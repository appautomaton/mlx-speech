"""Non-streaming inference helpers for MossTTSDelay."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from ..models.moss_delay import MossTTSDelayModel, MossTTSDelayProcessor
from .moss_local import DEFAULT_SAFETY_MAX_NEW_TOKENS


TTSD_DEFAULT_TEMPERATURE = 1.1
TTSD_DEFAULT_TOP_P = 0.9
TTSD_DEFAULT_TOP_K = 50
TTSD_DEFAULT_REPETITION_PENALTY = 1.1
TTSD_DEFAULT_MAX_NEW_TOKENS = 2048


@dataclass(frozen=True)
class MossTTSDelayGenerationConfig:
    """Sampling controls for the uncached MossTTSDelay generation loop."""

    use_kv_cache: bool = True
    max_new_tokens: int | None = TTSD_DEFAULT_MAX_NEW_TOKENS
    safety_max_new_tokens: int = DEFAULT_SAFETY_MAX_NEW_TOKENS
    text_temperature: float = 1.5
    text_top_k: int | None = 50
    text_top_p: float | None = 1.0
    text_repetition_penalty: float = 1.0
    audio_temperature: float = TTSD_DEFAULT_TEMPERATURE
    audio_top_k: int | None = TTSD_DEFAULT_TOP_K
    audio_top_p: float | None = TTSD_DEFAULT_TOP_P
    audio_repetition_penalty: float = TTSD_DEFAULT_REPETITION_PENALTY
    do_sample: bool | None = None


@dataclass(frozen=True)
class MossTTSDelayGenerationOutput:
    sequences: mx.array
    generated_rows: mx.array
    messages: tuple[tuple[int, mx.array], ...]
    stop_reached: bool


@dataclass(frozen=True)
class MossTTSDelaySynthesisOutput:
    generation: MossTTSDelayGenerationOutput
    waveform: mx.array
    sample_rate: int
    content: str | None = None
    audio_segments: tuple[mx.array, ...] = ()


@dataclass(frozen=True)
class MossTTSDelayBatchSynthesisOutput:
    generation: MossTTSDelayGenerationOutput
    outputs: tuple[MossTTSDelaySynthesisOutput, ...]
    sample_rate: int


def _resolve_max_new_tokens(config: MossTTSDelayGenerationConfig) -> int:
    if config.max_new_tokens is None:
        return int(config.safety_max_new_tokens)
    return min(int(config.max_new_tokens), int(config.safety_max_new_tokens))


def _can_use_kv_cache(
    model: MossTTSDelayModel,
    input_ids: mx.array,
    config: MossTTSDelayGenerationConfig,
) -> bool:
    if not config.use_kv_cache:
        return False
    if not hasattr(model, "prefill") or not hasattr(model, "decode_step"):
        return False
    return int(input_ids.shape[0]) == 1


def _find_last_equal_c(values: mx.array, target: int) -> list[int]:
    rows = values.tolist()
    result: list[int] = []
    for row in rows:
        index = -1
        for candidate_index, value in enumerate(row):
            if int(value) == int(target):
                index = candidate_index
        result.append(index)
    return result


def _resolve_do_sample(temperature: float, override: bool | None) -> tuple[float, bool]:
    if temperature <= 0.0:
        return 1.0, False
    if override is None:
        do_sample = temperature > 0.0
    else:
        do_sample = bool(override)
    if not do_sample:
        return 1.0, False
    return float(temperature), True


def _negative_inf(dtype: mx.Dtype) -> mx.array:
    return mx.array(float("-inf"), dtype=dtype)


def _apply_top_k_delay(logits: mx.array, top_k: int | None) -> mx.array:
    if top_k is None or top_k <= 0 or top_k >= int(logits.shape[-1]):
        return logits
    top_k_indices = mx.argsort(-logits, axis=-1)[..., :top_k]
    top_k_values = mx.take_along_axis(logits, top_k_indices, axis=-1)
    neg_inf = _negative_inf(logits.dtype)
    filtered_logits = mx.full(logits.shape, neg_inf, dtype=logits.dtype)
    return mx.put_along_axis(filtered_logits, top_k_indices, top_k_values, axis=-1)


def _apply_top_p_delay(logits: mx.array, top_p: float | None) -> mx.array:
    if top_p is None or top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be > 0 when provided.")

    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    to_remove = cumulative_probs > top_p
    to_remove = mx.concatenate(
        [mx.zeros_like(to_remove[..., :1]), to_remove[..., :-1]],
        axis=-1,
    )
    neg_inf = _negative_inf(logits.dtype)
    filtered_sorted_logits = mx.where(to_remove, neg_inf, sorted_logits)
    filtered_logits = mx.full(logits.shape, neg_inf, dtype=logits.dtype)
    return mx.put_along_axis(filtered_logits, sorted_indices, filtered_sorted_logits, axis=-1)


def _apply_repetition_penalty_delay_pattern(
    logits: mx.array,
    previous_tokens: mx.array | None,
    penalty: float,
) -> mx.array:
    if previous_tokens is None or penalty == 1.0:
        return logits

    adjusted = logits
    if logits.ndim == 2:
        seen_tokens = sorted({int(token) for token in previous_tokens.reshape(-1).tolist()})
        if not seen_tokens:
            return adjusted

        token_indices = mx.array(seen_tokens, dtype=mx.int32)[None, :]
        token_indices = mx.broadcast_to(token_indices, (int(adjusted.shape[0]), len(seen_tokens)))
        values = mx.take_along_axis(adjusted, token_indices, axis=-1)
        updated = mx.where(values > 0, values / penalty, values * penalty)
        return mx.put_along_axis(adjusted, token_indices, updated, axis=-1)

    if logits.ndim != 3:
        raise ValueError(f"Expected delay logits with rank 2 or 3, got {logits.shape}.")

    for head_index in range(int(logits.shape[1])):
        seen_tokens = sorted(
            {int(token) for token in previous_tokens[..., head_index].reshape(-1).tolist()}
        )
        if not seen_tokens:
            continue

        token_indices = mx.array(seen_tokens, dtype=mx.int32)[None, :]
        token_indices = mx.broadcast_to(token_indices, (int(adjusted.shape[0]), len(seen_tokens)))
        head_values = mx.take_along_axis(adjusted[:, head_index, :], token_indices, axis=-1)
        updated = mx.where(
            head_values > 0,
            head_values / penalty,
            head_values * penalty,
        )
        adjusted_head = mx.put_along_axis(
            adjusted[:, head_index, :],
            token_indices,
            updated,
            axis=-1,
        )
        adjusted = mx.concatenate(
            [
                adjusted[:, :head_index, :],
                adjusted_head[:, None, :],
                adjusted[:, head_index + 1 :, :],
            ],
            axis=1,
        )
    return adjusted


def _sample_delay_token(
    logits: mx.array,
    *,
    previous_tokens: mx.array | None,
    repetition_penalty: float,
    top_p: float | None,
    top_k: int | None,
    do_sample: bool,
) -> mx.array:
    warped = _apply_repetition_penalty_delay_pattern(
        logits.astype(mx.float32),
        previous_tokens,
        repetition_penalty,
    )

    if not do_sample:
        return mx.argmax(warped, axis=-1).astype(mx.int32)

    original_shape = warped.shape
    reshaped = warped.reshape(-1, int(warped.shape[-1]))
    reshaped = _apply_top_k_delay(reshaped, top_k)
    reshaped = _apply_top_p_delay(reshaped, top_p)
    sampled = mx.random.categorical(reshaped, axis=-1).astype(mx.int32)
    return sampled.reshape(original_shape[:-1])


def _allocate_delay_sequence_buffer(
    input_ids: mx.array,
    *,
    max_new_tokens: int,
    pad_code: int,
) -> tuple[mx.array, int]:
    batch_size, prompt_length, channels = input_ids.shape
    total_length = prompt_length + max_new_tokens
    sequences = mx.full(
        (batch_size, total_length, channels),
        pad_code,
        dtype=mx.int32,
    )
    sequences[:, :prompt_length, :] = input_ids.astype(mx.int32)
    return sequences, int(prompt_length)


def _initialize_delay_state(
    input_ids: mx.array,
    *,
    audio_start_token_id: int,
    audio_assistant_gen_slot_token_id: int,
    max_int64: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    batch_size, seq_len, _ = input_ids.shape
    last_text_tokens = input_ids[:, -1, 0].astype(mx.int32)
    audio_start_indices = mx.array(
        _find_last_equal_c(input_ids[:, :, 0], audio_start_token_id),
        dtype=mx.int64,
    )
    is_continuation = (last_text_tokens == audio_start_token_id) | (
        last_text_tokens == audio_assistant_gen_slot_token_id
    )
    has_audio_start = audio_start_indices != -1
    prompt_lengths = mx.full((int(batch_size),), int(seq_len), dtype=mx.int64)
    audio_lengths = mx.where(
        is_continuation & has_audio_start,
        prompt_lengths - audio_start_indices,
        mx.zeros((int(batch_size),), dtype=mx.int64),
    )
    delayed_lengths = mx.full((int(batch_size),), max_int64, dtype=mx.int64)
    is_audio = is_continuation & has_audio_start
    is_stopping = mx.zeros((int(batch_size),), dtype=mx.bool_)
    return is_stopping, audio_lengths, delayed_lengths, is_audio


def _mask_delay_text_logits(
    text_logits: mx.array,
    *,
    is_audio: mx.array,
    time_step: int,
    n_vq: int,
    pad_token_id: int,
    audio_assistant_gen_slot_token_id: int,
    audio_assistant_delay_slot_token_id: int,
    audio_end_token_id: int,
    im_end_token_id: int,
) -> mx.array:
    batch_size = int(text_logits.shape[0])
    neg_inf = _negative_inf(text_logits.dtype)

    forbidden_indices = mx.array(
        [
            pad_token_id,
            audio_assistant_gen_slot_token_id,
            audio_assistant_delay_slot_token_id,
            audio_end_token_id,
        ],
        dtype=mx.int32,
    )[None, :]
    forbidden_indices = mx.broadcast_to(forbidden_indices, (batch_size, 4))
    forbidden_values = mx.full((batch_size, 4), neg_inf, dtype=text_logits.dtype)
    not_audio_logits = mx.put_along_axis(
        text_logits,
        forbidden_indices,
        forbidden_values,
        axis=-1,
    )

    allowed_indices = mx.array(
        [
            audio_assistant_gen_slot_token_id,
            audio_assistant_delay_slot_token_id,
        ],
        dtype=mx.int32,
    )[None, :]
    allowed_indices = mx.broadcast_to(allowed_indices, (batch_size, 2))
    allowed_values = mx.take_along_axis(text_logits, allowed_indices, axis=-1)
    audio_logits = mx.full(text_logits.shape, neg_inf, dtype=text_logits.dtype)
    audio_logits = mx.put_along_axis(
        audio_logits,
        allowed_indices,
        allowed_values,
        axis=-1,
    )

    masked = mx.where(is_audio[:, None], audio_logits, not_audio_logits)
    if time_step == 0:
        delay_indices = mx.full(
            (batch_size, 1),
            audio_assistant_delay_slot_token_id,
            dtype=mx.int32,
        )
        masked = mx.put_along_axis(
            masked,
            delay_indices,
            mx.full((batch_size, 1), neg_inf, dtype=text_logits.dtype),
            axis=-1,
        )
    if time_step <= n_vq:
        stop_indices = mx.full((batch_size, 1), im_end_token_id, dtype=mx.int32)
        masked = mx.put_along_axis(
            masked,
            stop_indices,
            mx.full((batch_size, 1), neg_inf, dtype=text_logits.dtype),
            axis=-1,
        )
    return masked


def _build_delay_sampling_audio_mask(
    *,
    audio_lengths: mx.array,
    delayed_lengths: mx.array,
    is_stopping: mx.array,
    n_vq: int,
    max_int64: int,
) -> mx.array:
    channel_indices = mx.arange(n_vq, dtype=mx.int64)[None, :]
    pre_audio_mask = audio_lengths[:, None] > channel_indices
    post_audio_mask = mx.where(
        delayed_lengths[:, None] == max_int64,
        mx.ones((int(audio_lengths.shape[0]), n_vq), dtype=mx.bool_),
        channel_indices > (delayed_lengths[:, None] - 1),
    )
    return pre_audio_mask & post_audio_mask & (~is_stopping[:, None])


def _update_delay_state(
    next_text_tokens: mx.array,
    *,
    audio_lengths: mx.array,
    delayed_lengths: mx.array,
    is_audio: mx.array,
    is_stopping: mx.array,
    audio_start_token_id: int,
    audio_end_token_id: int,
    audio_assistant_gen_slot_token_id: int,
    audio_assistant_delay_slot_token_id: int,
    im_end_token_id: int,
    n_vq: int,
    max_int64: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    audio_like = (
        (next_text_tokens == audio_start_token_id)
        | (next_text_tokens == audio_assistant_gen_slot_token_id)
        | (next_text_tokens == audio_assistant_delay_slot_token_id)
    )
    next_audio_lengths = mx.where(audio_like, audio_lengths + 1, audio_lengths)
    next_audio_lengths = mx.where(next_text_tokens == audio_end_token_id, 0, next_audio_lengths)

    started_delay = (delayed_lengths == max_int64) & (
        next_text_tokens == audio_assistant_delay_slot_token_id
    )
    continuing_delay = delayed_lengths != max_int64
    incremented_delays = delayed_lengths + 1
    incremented_delays = mx.where(
        incremented_delays > n_vq,
        mx.full(delayed_lengths.shape, max_int64, dtype=delayed_lengths.dtype),
        incremented_delays,
    )
    next_delayed_lengths = mx.where(started_delay, 0, delayed_lengths)
    next_delayed_lengths = mx.where(
        continuing_delay,
        incremented_delays,
        next_delayed_lengths,
    )

    next_is_audio = mx.where(next_text_tokens == audio_end_token_id, False, is_audio)
    next_is_audio = next_is_audio | (next_text_tokens == audio_start_token_id)
    next_is_stopping = is_stopping | (next_text_tokens == im_end_token_id)
    return next_audio_lengths, next_delayed_lengths, next_is_audio, next_is_stopping


def _generate_moss_tts_delay_uncached(
    model: MossTTSDelayModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    config: MossTTSDelayGenerationConfig | None = None,
) -> MossTTSDelayGenerationOutput:
    config = config or MossTTSDelayGenerationConfig()
    max_new_tokens = _resolve_max_new_tokens(config)
    text_temperature, text_do_sample = _resolve_do_sample(
        config.text_temperature,
        config.do_sample,
    )
    audio_temperature, audio_do_sample = _resolve_do_sample(
        config.audio_temperature,
        config.do_sample,
    )

    batch_size, seq_len, channels = input_ids.shape
    n_vq = channels - 1
    generation_ids = input_ids.astype(mx.int32)
    current_attention_mask = attention_mask.astype(mx.bool_)

    pad_token_id = int(model.config.pad_token_id)
    audio_pad_code = int(model.config.audio_pad_code)
    audio_start_token_id = int(model.config.audio_start_token_id)
    audio_end_token_id = int(model.config.audio_end_token_id)
    audio_assistant_gen_slot_token_id = int(model.config.audio_assistant_gen_slot_token_id)
    audio_assistant_delay_slot_token_id = int(model.config.audio_assistant_delay_slot_token_id)
    im_start_token_id = int(model.config.im_start_token_id)
    im_end_token_id = int(model.config.im_end_token_id)
    max_int64 = (1 << 63) - 1

    last_text_tokens = [int(token) for token in input_ids[:, -1, 0].tolist()]
    audio_start_indices = _find_last_equal_c(input_ids[:, :, 0], audio_start_token_id)

    is_stopping = [False] * int(batch_size)
    audio_lengths = [0] * int(batch_size)
    delayed_lengths = [max_int64] * int(batch_size)
    is_audio = [False] * int(batch_size)

    for batch_index in range(int(batch_size)):
        is_continuation = last_text_tokens[batch_index] in {
            audio_start_token_id,
            audio_assistant_gen_slot_token_id,
        }
        if is_continuation and audio_start_indices[batch_index] != -1:
            audio_lengths[batch_index] = int(seq_len) - int(audio_start_indices[batch_index])
            is_audio[batch_index] = True

    stop_reached = False
    for time_step in range(max_new_tokens):
        outputs = model(
            input_ids=generation_ids,
            attention_mask=current_attention_mask,
            output_hidden_states=False,
        )
        next_token_logits = [
            (logits[:, -1, :] / text_temperature).astype(mx.float32)
            if logit_index == 0
            else (logits[:, -1, :] / audio_temperature).astype(mx.float32)
            for logit_index, logits in enumerate(outputs.logits_all)
        ]

        next_text_tokens: list[int] = [pad_token_id] * int(batch_size)
        next_audio_tokens = [[audio_pad_code] * n_vq for _ in range(int(batch_size))]

        for batch_index in range(int(batch_size)):
            if is_stopping[batch_index]:
                continue

            if delayed_lengths[batch_index] < n_vq:
                next_text_tokens[batch_index] = audio_assistant_delay_slot_token_id
            elif delayed_lengths[batch_index] == n_vq:
                next_text_tokens[batch_index] = audio_end_token_id
                is_audio[batch_index] = False
            else:
                text_logits = mx.array(next_token_logits[0][batch_index])
                if not is_audio[batch_index]:
                    for token_id in (
                        pad_token_id,
                        audio_assistant_gen_slot_token_id,
                        audio_assistant_delay_slot_token_id,
                        audio_end_token_id,
                    ):
                        text_logits[token_id] = _negative_inf(text_logits.dtype)
                else:
                    allowed = {
                        audio_assistant_gen_slot_token_id,
                        audio_assistant_delay_slot_token_id,
                    }
                    for token_id in range(int(text_logits.shape[0])):
                        if token_id not in allowed:
                            text_logits[token_id] = _negative_inf(text_logits.dtype)
                if time_step == 0:
                    text_logits[audio_assistant_delay_slot_token_id] = _negative_inf(
                        text_logits.dtype
                    )
                if time_step <= n_vq:
                    text_logits[im_end_token_id] = _negative_inf(text_logits.dtype)

                next_text_token = int(
                    _sample_delay_token(
                        text_logits[None, :],
                        previous_tokens=None,
                        repetition_penalty=1.0,
                        top_p=config.text_top_p,
                        top_k=config.text_top_k,
                        do_sample=text_do_sample,
                    )[0].item()
                )
                next_text_tokens[batch_index] = next_text_token
                if next_text_token == audio_start_token_id:
                    is_audio[batch_index] = True
                if next_text_token == im_end_token_id:
                    is_stopping[batch_index] = True

        pre_audio_mask = mx.array(
            [
                [audio_lengths[batch_index] > channel_index for channel_index in range(n_vq)]
                for batch_index in range(int(batch_size))
            ],
            dtype=mx.bool_,
        )
        post_audio_mask = mx.array(
            [
                [
                    True
                    if delayed_lengths[batch_index] == max_int64
                    else channel_index > delayed_lengths[batch_index] - 1
                    for channel_index in range(n_vq)
                ]
                for batch_index in range(int(batch_size))
            ],
            dtype=mx.bool_,
        )
        sampling_audio_mask = pre_audio_mask & post_audio_mask

        if bool(mx.any(sampling_audio_mask[:, 0]).item()):
            channel0_indices = [
                batch_index
                for batch_index, keep in enumerate(sampling_audio_mask[:, 0].tolist())
                if bool(keep)
            ]
            channel0_logits = mx.array(next_token_logits[1][mx.array(channel0_indices, dtype=mx.int32)])
            channel0_logits[:, audio_pad_code] = _negative_inf(channel0_logits.dtype)
            channel0_tokens = _sample_delay_token(
                channel0_logits,
                previous_tokens=generation_ids[:, :, 1],
                repetition_penalty=config.audio_repetition_penalty,
                top_p=config.audio_top_p,
                top_k=config.audio_top_k,
                do_sample=audio_do_sample,
            )
            for output_index, batch_index in enumerate(channel0_indices):
                next_audio_tokens[batch_index][0] = int(channel0_tokens[output_index].item())

        if n_vq > 1:
            tail_mask = sampling_audio_mask[:, 1:]
            if bool(mx.any(tail_mask).item()):
                stacked_logits = mx.stack(next_token_logits[2:], axis=1)
                flat_logits = stacked_logits.reshape(-1, int(stacked_logits.shape[-1]))
                selected_indices = [
                    index
                    for index, keep in enumerate(tail_mask.reshape(-1).tolist())
                    if bool(keep)
                ]
                selected_logits = mx.array(flat_logits[mx.array(selected_indices, dtype=mx.int32)])
                selected_logits[:, audio_pad_code] = _negative_inf(selected_logits.dtype)
                selected_tokens = _sample_delay_token(
                    selected_logits,
                    previous_tokens=generation_ids[:, :, 2:],
                    repetition_penalty=config.audio_repetition_penalty,
                    top_p=config.audio_top_p,
                    top_k=config.audio_top_k,
                    do_sample=audio_do_sample,
                )
                tail_positions = [
                    (batch_index, channel_index)
                    for batch_index in range(int(batch_size))
                    for channel_index in range(1, n_vq)
                    if bool(tail_mask[batch_index, channel_index - 1].item())
                ]
                for output_index, (batch_index, channel_index) in enumerate(tail_positions):
                    next_audio_tokens[batch_index][channel_index] = int(
                        selected_tokens[output_index].item()
                    )

        for batch_index, next_text_token in enumerate(next_text_tokens):
            if next_text_token in {
                audio_start_token_id,
                audio_assistant_gen_slot_token_id,
                audio_assistant_delay_slot_token_id,
            }:
                audio_lengths[batch_index] += 1
            if next_text_token == audio_end_token_id:
                audio_lengths[batch_index] = 0
            if (
                delayed_lengths[batch_index] == max_int64
                and next_text_token == audio_assistant_delay_slot_token_id
            ):
                delayed_lengths[batch_index] = 0
            elif delayed_lengths[batch_index] != max_int64:
                delayed_lengths[batch_index] += 1
                if delayed_lengths[batch_index] > n_vq:
                    delayed_lengths[batch_index] = max_int64

        new_rows = mx.array(
            [
                [next_text_tokens[batch_index], *next_audio_tokens[batch_index]]
                for batch_index in range(int(batch_size))
            ],
            dtype=mx.int32,
        )[:, None, :]
        generation_ids = mx.concatenate([generation_ids, new_rows], axis=1)
        current_attention_mask = mx.concatenate(
            [
                current_attention_mask,
                mx.array([[not state] for state in is_stopping], dtype=mx.bool_),
            ],
            axis=1,
        )

        if all(is_stopping):
            stop_reached = True
            break

    start_indices = [index + 3 for index in _find_last_equal_c(input_ids[:, :, 0], im_start_token_id)]
    start_lengths = [int(seq_len) - index for index in start_indices]
    messages: list[tuple[int, mx.array]] = []
    for batch_index, (start_index, start_length) in enumerate(zip(start_indices, start_lengths)):
        messages.append((start_length, generation_ids[batch_index, start_index:, :]))

    return MossTTSDelayGenerationOutput(
        sequences=generation_ids,
        generated_rows=generation_ids[:, seq_len:, :],
        messages=tuple(messages),
        stop_reached=stop_reached,
    )


def _generate_moss_tts_delay_cached(
    model: MossTTSDelayModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    config: MossTTSDelayGenerationConfig | None = None,
) -> MossTTSDelayGenerationOutput:
    config = config or MossTTSDelayGenerationConfig()
    max_new_tokens = _resolve_max_new_tokens(config)
    text_temperature, text_do_sample = _resolve_do_sample(
        config.text_temperature,
        config.do_sample,
    )
    audio_temperature, audio_do_sample = _resolve_do_sample(
        config.audio_temperature,
        config.do_sample,
    )

    batch_size, seq_len, channels = input_ids.shape
    if batch_size != 1:
        raise ValueError("Cached TTSD generation currently supports batch_size == 1 only.")
    n_vq = channels - 1

    pad_token_id = int(model.config.pad_token_id)
    audio_pad_code = int(model.config.audio_pad_code)
    audio_start_token_id = int(model.config.audio_start_token_id)
    audio_end_token_id = int(model.config.audio_end_token_id)
    audio_assistant_gen_slot_token_id = int(model.config.audio_assistant_gen_slot_token_id)
    audio_assistant_delay_slot_token_id = int(model.config.audio_assistant_delay_slot_token_id)
    im_start_token_id = int(model.config.im_start_token_id)
    im_end_token_id = int(model.config.im_end_token_id)
    max_int64 = (1 << 63) - 1

    sequences, current_length = _allocate_delay_sequence_buffer(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_code=audio_pad_code,
    )
    (
        is_stopping,
        audio_lengths,
        delayed_lengths,
        is_audio,
    ) = _initialize_delay_state(
        input_ids,
        audio_start_token_id=audio_start_token_id,
        audio_assistant_gen_slot_token_id=audio_assistant_gen_slot_token_id,
        max_int64=max_int64,
    )

    outputs, kv_cache = model.prefill(
        input_ids=input_ids,
        attention_mask=attention_mask.astype(mx.bool_),
        max_cache_len=seq_len + max_new_tokens,
    )

    stop_reached = False
    for time_step in range(max_new_tokens):
        next_token_logits = [
            (logits[:, -1, :] / text_temperature).astype(mx.float32)
            if logit_index == 0
            else (logits[:, -1, :] / audio_temperature).astype(mx.float32)
            for logit_index, logits in enumerate(outputs.logits_all)
        ]
        forced_delay = delayed_lengths < n_vq
        forced_end = delayed_lengths == n_vq

        masked_text_logits = _mask_delay_text_logits(
            next_token_logits[0],
            is_audio=is_audio,
            time_step=time_step,
            n_vq=n_vq,
            pad_token_id=pad_token_id,
            audio_assistant_gen_slot_token_id=audio_assistant_gen_slot_token_id,
            audio_assistant_delay_slot_token_id=audio_assistant_delay_slot_token_id,
            audio_end_token_id=audio_end_token_id,
            im_end_token_id=im_end_token_id,
        )
        sampled_text_tokens = _sample_delay_token(
            masked_text_logits,
            previous_tokens=None,
            repetition_penalty=1.0,
            top_p=config.text_top_p,
            top_k=config.text_top_k,
            do_sample=text_do_sample,
        )
        next_text_tokens = mx.where(
            forced_delay,
            mx.full((batch_size,), audio_assistant_delay_slot_token_id, dtype=mx.int32),
            sampled_text_tokens,
        )
        next_text_tokens = mx.where(
            forced_end,
            mx.full((batch_size,), audio_end_token_id, dtype=mx.int32),
            next_text_tokens,
        )
        next_text_tokens = mx.where(
            is_stopping,
            mx.full((batch_size,), pad_token_id, dtype=mx.int32),
            next_text_tokens,
        )
        is_audio = mx.where(next_text_tokens == audio_end_token_id, False, is_audio)
        is_audio = is_audio | (next_text_tokens == audio_start_token_id)
        is_stopping = is_stopping | (next_text_tokens == im_end_token_id)

        sampling_audio_mask = _build_delay_sampling_audio_mask(
            audio_lengths=audio_lengths,
            delayed_lengths=delayed_lengths,
            is_stopping=is_stopping,
            n_vq=n_vq,
            max_int64=max_int64,
        )

        next_audio_tokens = mx.full((batch_size, n_vq), audio_pad_code, dtype=mx.int32)
        channel0_logits = mx.array(next_token_logits[1])
        channel0_logits[:, audio_pad_code] = _negative_inf(channel0_logits.dtype)
        channel0_tokens = _sample_delay_token(
            channel0_logits,
            previous_tokens=sequences[:, :current_length, 1],
            repetition_penalty=config.audio_repetition_penalty,
            top_p=config.audio_top_p,
            top_k=config.audio_top_k,
            do_sample=audio_do_sample,
        )
        next_audio_tokens[:, 0] = mx.where(
            sampling_audio_mask[:, 0],
            channel0_tokens,
            next_audio_tokens[:, 0],
        )

        if n_vq > 1:
            tail_logits = mx.stack(next_token_logits[2:], axis=1)
            tail_logits[..., audio_pad_code] = _negative_inf(tail_logits.dtype)
            flat_tail_logits = tail_logits.reshape(-1, int(tail_logits.shape[-1]))
            tail_tokens = _sample_delay_token(
                flat_tail_logits,
                previous_tokens=sequences[:, :current_length, 2:],
                repetition_penalty=config.audio_repetition_penalty,
                top_p=config.audio_top_p,
                top_k=config.audio_top_k,
                do_sample=audio_do_sample,
            ).reshape(batch_size, n_vq - 1)
            next_audio_tokens[:, 1:] = mx.where(
                sampling_audio_mask[:, 1:],
                tail_tokens,
                next_audio_tokens[:, 1:],
            )

        (
            audio_lengths,
            delayed_lengths,
            is_audio,
            is_stopping,
        ) = _update_delay_state(
            next_text_tokens,
            audio_lengths=audio_lengths,
            delayed_lengths=delayed_lengths,
            is_audio=is_audio,
            is_stopping=is_stopping,
            audio_start_token_id=audio_start_token_id,
            audio_end_token_id=audio_end_token_id,
            audio_assistant_gen_slot_token_id=audio_assistant_gen_slot_token_id,
            audio_assistant_delay_slot_token_id=audio_assistant_delay_slot_token_id,
            im_end_token_id=im_end_token_id,
            n_vq=n_vq,
            max_int64=max_int64,
        )

        next_rows = mx.concatenate([next_text_tokens[:, None], next_audio_tokens], axis=1)
        sequences[:, current_length, :] = next_rows
        current_length += 1

        if bool(mx.all(is_stopping).item()):
            stop_reached = True
            break

        outputs = model.decode_step(
            input_ids=next_rows[:, None, :],
            kv_cache=kv_cache,
        )

    start_indices = [index + 3 for index in _find_last_equal_c(input_ids[:, :, 0], im_start_token_id)]
    start_lengths = [int(seq_len) - index for index in start_indices]
    sequences = sequences[:, :current_length, :]
    messages: list[tuple[int, mx.array]] = []
    for batch_index, (start_index, start_length) in enumerate(zip(start_indices, start_lengths)):
        messages.append((start_length, sequences[batch_index, start_index:, :]))

    return MossTTSDelayGenerationOutput(
        sequences=sequences,
        generated_rows=sequences[:, seq_len:current_length, :],
        messages=tuple(messages),
        stop_reached=stop_reached,
    )


def generate_moss_tts_delay(
    model: MossTTSDelayModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    config: MossTTSDelayGenerationConfig | None = None,
) -> MossTTSDelayGenerationOutput:
    config = config or MossTTSDelayGenerationConfig()
    if _can_use_kv_cache(model, input_ids, config):
        return _generate_moss_tts_delay_cached(
            model,
            input_ids,
            attention_mask,
            config=config,
        )
    return _generate_moss_tts_delay_uncached(
        model,
        input_ids,
        attention_mask,
        config=config,
    )


def synthesize_moss_tts_delay_conversations(
    model: MossTTSDelayModel,
    processor: MossTTSDelayProcessor,
    *,
    conversations: list[list[dict]],
    mode: str,
    config: MossTTSDelayGenerationConfig | None = None,
) -> MossTTSDelayBatchSynthesisOutput:
    batch = processor(conversations, mode=mode)
    generation = generate_moss_tts_delay(
        model,
        batch.input_ids,
        batch.attention_mask,
        config=config,
    )
    messages = processor.decode_sequences(list(generation.messages))

    outputs: list[MossTTSDelaySynthesisOutput] = []
    sample_rate = int(processor.model_config.sampling_rate)
    for message in messages:
        if message is None or len(message.audio_codes_list) == 0:
            waveform = mx.zeros((0,), dtype=mx.float32)
            audio_segments: tuple[mx.array, ...] = ()
            content = None if message is None else message.content
        else:
            audio_segments = tuple(audio.astype(mx.float32) for audio in message.audio_codes_list)
            waveform = mx.concatenate(list(audio_segments), axis=0) if audio_segments else mx.zeros((0,), dtype=mx.float32)
            content = message.content

        outputs.append(
            MossTTSDelaySynthesisOutput(
                generation=generation,
                waveform=waveform,
                sample_rate=sample_rate,
                content=content,
                audio_segments=audio_segments,
            )
        )

    return MossTTSDelayBatchSynthesisOutput(
        generation=generation,
        outputs=tuple(outputs),
        sample_rate=sample_rate,
    )
