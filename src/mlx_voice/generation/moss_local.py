"""Non-streaming inference helpers for MossTTSLocal."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx

from ..models.moss_audio_tokenizer import MossAudioTokenizerModel
from ..models.moss_local import (
    AssistantMessage,
    LocalKVCache,
    MossTTSLocalModel,
    MossTTSLocalProcessor,
    estimate_duration_tokens,
)
from ..models.moss_local.model import _linear_forward

MODEL_DEFAULT_AUDIO_TEMPERATURE = 1.0
MODEL_DEFAULT_AUDIO_TOP_K = 50
MODEL_DEFAULT_AUDIO_TOP_P = 0.95
MODEL_DEFAULT_AUDIO_REPETITION_PENALTY = 1.1
APP_DEFAULT_AUDIO_TEMPERATURE = 1.7
APP_DEFAULT_AUDIO_TOP_K = 25
APP_DEFAULT_AUDIO_TOP_P = 0.8
APP_DEFAULT_AUDIO_REPETITION_PENALTY = 1.0
DEFAULT_SAFETY_MAX_NEW_TOKENS = 2048


@dataclass(frozen=True)
class MossTTSLocalGenerationConfig:
    """Sampling controls for the MossTTSLocal generation loop."""

    max_new_tokens: int | None = 1024
    safety_max_new_tokens: int = DEFAULT_SAFETY_MAX_NEW_TOKENS
    n_vq_for_inference: int | None = None
    text_temperature: float = 1.5
    text_top_k: int | None = 50
    text_top_p: float | None = 1.0
    text_repetition_penalty: float = 1.0
    audio_temperature: float = MODEL_DEFAULT_AUDIO_TEMPERATURE
    audio_top_k: int | None = MODEL_DEFAULT_AUDIO_TOP_K
    audio_top_p: float | None = MODEL_DEFAULT_AUDIO_TOP_P
    audio_repetition_penalty: float = MODEL_DEFAULT_AUDIO_REPETITION_PENALTY
    do_sample: bool | None = None
    use_kv_cache: bool = True

    @classmethod
    def app_defaults(cls, **overrides: Any) -> "MossTTSLocalGenerationConfig":
        payload: dict[str, Any] = {
            "audio_temperature": APP_DEFAULT_AUDIO_TEMPERATURE,
            "audio_top_k": APP_DEFAULT_AUDIO_TOP_K,
            "audio_top_p": APP_DEFAULT_AUDIO_TOP_P,
            "audio_repetition_penalty": APP_DEFAULT_AUDIO_REPETITION_PENALTY,
        }
        payload.update(overrides)
        return cls(**payload)


@dataclass(frozen=True)
class MossTTSLocalGenerationOutput:
    sequences: mx.array
    generated_rows: mx.array
    audio_codes_list: tuple[mx.array, ...]
    stop_reached: bool


@dataclass(frozen=True)
class MossTTSLocalSynthesisOutput:
    generation: MossTTSLocalGenerationOutput
    waveform: mx.array
    sample_rate: int
    content: str | None = None
    audio_segments: tuple[mx.array, ...] = ()


@dataclass(frozen=True)
class MossTTSLocalBatchSynthesisOutput:
    generation: MossTTSLocalGenerationOutput
    outputs: tuple[MossTTSLocalSynthesisOutput, ...]
    sample_rate: int


def _resolve_sampling_config(
    layer_index: int,
    config: MossTTSLocalGenerationConfig,
) -> tuple[float, int | None, float | None, float, bool]:
    if layer_index == 0:
        temperature = config.text_temperature
        top_k = config.text_top_k
        top_p = config.text_top_p
        repetition_penalty = config.text_repetition_penalty
    else:
        temperature = config.audio_temperature
        top_k = config.audio_top_k
        top_p = config.audio_top_p
        repetition_penalty = config.audio_repetition_penalty

    if config.do_sample is None:
        do_sample = temperature > 0.0
    else:
        do_sample = bool(config.do_sample)

    if not do_sample:
        temperature = 1.0

    return temperature, top_k, top_p, repetition_penalty, do_sample


def _apply_repetition_penalty(
    logits: mx.array,
    previous_tokens: mx.array | None,
    penalty: float,
) -> mx.array:
    if previous_tokens is None or penalty == 1.0:
        return logits

    adjusted = logits
    for batch_index in range(int(logits.shape[0])):
        seen = {int(token) for token in previous_tokens[batch_index].tolist()}
        for token in seen:
            value = adjusted[batch_index, token]
            adjusted[batch_index, token] = mx.where(
                value > 0,
                value / penalty,
                value * penalty,
            )
    return adjusted


def _apply_top_k(logits: mx.array, top_k: int | None) -> mx.array:
    if top_k is None or top_k <= 0 or top_k >= int(logits.shape[-1]):
        return logits
    kth_values = mx.topk(logits, k=top_k, axis=-1)[..., -1:]
    neg_inf = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
    return mx.where(logits < kth_values, neg_inf, logits)


def _apply_top_p(logits: mx.array, top_p: float | None) -> mx.array:
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
        [
            mx.zeros_like(to_remove[..., :1]),
            to_remove[..., :-1],
        ],
        axis=-1,
    )
    neg_inf = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
    filtered_sorted_logits = mx.where(to_remove, neg_inf, sorted_logits)
    filtered_logits = mx.full(logits.shape, neg_inf, dtype=logits.dtype)
    return mx.put_along_axis(filtered_logits, sorted_indices, filtered_sorted_logits, axis=-1)


def sample_next_token(
    logits: mx.array,
    *,
    previous_tokens: mx.array | None,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float,
    do_sample: bool,
) -> mx.array:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")

    if not do_sample:
        return mx.argmax(logits.astype(mx.float32), axis=-1).astype(mx.int32)

    warped = logits.astype(mx.float32)
    warped = _apply_repetition_penalty(warped, previous_tokens, repetition_penalty)
    warped = warped / temperature
    warped = _apply_top_k(warped, top_k)
    warped = _apply_top_p(warped, top_p)
    return mx.random.categorical(warped, axis=-1).astype(mx.int32)


def extract_audio_code_sequences(
    sequences: mx.array,
    *,
    prompt_length: int,
    pad_code: int,
    n_vq: int,
    stop_token: int | None = None,
) -> tuple[mx.array, ...]:
    extracted: list[mx.array] = []
    generated_rows = sequences[:, prompt_length:, :]
    for batch_index in range(int(generated_rows.shape[0])):
        rows = generated_rows[batch_index]
        if rows.shape[0] == 0:
            extracted.append(mx.zeros((0, n_vq), dtype=mx.int32))
            continue
        if stop_token is not None:
            stop_indices = [
                idx for idx, token in enumerate(rows[:, 0].tolist()) if int(token) == stop_token
            ]
            if stop_indices:
                rows = rows[: stop_indices[0]]
                if rows.shape[0] == 0:
                    extracted.append(mx.zeros((0, n_vq), dtype=mx.int32))
                    continue
        audio_rows = rows[:, 1 : 1 + n_vq]
        keep_mask = ~mx.all(audio_rows == pad_code, axis=-1)
        keep_indices = [idx for idx, keep in enumerate(keep_mask.tolist()) if bool(keep)]
        if keep_indices:
            kept = audio_rows[mx.array(keep_indices, dtype=mx.int32)]
        else:
            kept = mx.zeros((0, n_vq), dtype=mx.int32)
        extracted.append(kept.astype(mx.int32))
    return tuple(extracted)


def generate_moss_tts_local(
    model: MossTTSLocalModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    config: MossTTSLocalGenerationConfig | None = None,
) -> MossTTSLocalGenerationOutput:
    config = config or MossTTSLocalGenerationConfig()
    if _can_use_kv_cache(input_ids, config):
        return _generate_moss_tts_local_cached(
            model,
            input_ids,
            attention_mask,
            config=config,
        )
    return _generate_moss_tts_local_uncached(
        model,
        input_ids,
        attention_mask,
        config=config,
    )


def _can_use_kv_cache(
    input_ids: mx.array,
    config: MossTTSLocalGenerationConfig,
) -> bool:
    if not config.use_kv_cache or int(input_ids.shape[0]) != 1:
        return False
    # Keep deterministic greedy/debug runs on the trusted full-sequence path.
    return config.do_sample is not False


def _resolve_generation_limit(config: MossTTSLocalGenerationConfig) -> int:
    if config.max_new_tokens is None:
        limit = int(config.safety_max_new_tokens)
    else:
        limit = int(config.max_new_tokens)
    if limit <= 0:
        raise ValueError("Generation limit must be > 0.")
    return limit


def _allocate_sequence_buffer(
    input_ids: mx.array,
    *,
    max_new_tokens: int,
) -> tuple[mx.array, int]:
    batch_size, prompt_length, channels = input_ids.shape
    total_length = prompt_length + max_new_tokens
    sequences = mx.zeros((batch_size, total_length, channels), dtype=mx.int32)
    sequences[:, :prompt_length, :] = input_ids.astype(mx.int32)
    return sequences, prompt_length


def _allocate_attention_mask_buffer(
    attention_mask: mx.array,
    *,
    max_new_tokens: int,
) -> tuple[mx.array, int]:
    batch_size, prompt_length = attention_mask.shape
    total_length = prompt_length + max_new_tokens
    full_mask = mx.zeros((batch_size, total_length), dtype=mx.bool_)
    full_mask[:, :prompt_length] = attention_mask.astype(mx.bool_)
    return full_mask, prompt_length


def _allocate_local_input_buffer(
    *,
    batch_size: int,
    max_steps: int,
    hidden_size: int,
    dtype: mx.Dtype,
) -> mx.array:
    return mx.zeros((batch_size, max_steps, hidden_size), dtype=dtype)


def _run_local_depth_full(
    model: MossTTSLocalModel,
    *,
    global_hidden: mx.array,
    sequences: mx.array,
    current_length: int,
    n_vq_for_inference: int,
    config: MossTTSLocalGenerationConfig,
) -> mx.array:
    batch_size = int(global_hidden.shape[0])
    channels = model.config.channels
    max_depth = min(channels, 1 + n_vq_for_inference)
    local_inputs = _allocate_local_input_buffer(
        batch_size=batch_size,
        max_steps=max_depth,
        hidden_size=model.config.local_hidden_size,
        dtype=global_hidden.dtype,
    )
    local_length = 0
    current_local_input = model.project_global_to_local(global_hidden)
    next_row = mx.full((batch_size, channels), model.config.audio_pad_code, dtype=mx.int32)

    for layer_index in range(max_depth):
        local_inputs[:, local_length : local_length + 1, :] = current_local_input[:, None, :]
        local_length += 1
        local_output = model.forward_local_sequence(local_inputs[:, :local_length, :])
        projected = model.local_to_speech_embedding_mlps[layer_index](
            local_output.last_hidden_state
        )
        projected = model.layer_norm_before_lm_heads[layer_index](projected)
        logits = _linear_forward(model.lm_heads[layer_index], projected[:, -1, :])
        if layer_index != 0:
            logits[:, model.config.audio_pad_code] = mx.array(
                mx.finfo(logits.dtype).min,
                dtype=logits.dtype,
            )
        temperature, top_k, top_p, repetition_penalty, do_sample = _resolve_sampling_config(
            layer_index,
            config,
        )
        previous_tokens = None
        if repetition_penalty != 1.0:
            previous_tokens = sequences[:, :current_length, layer_index]
        token = sample_next_token(
            logits,
            previous_tokens=previous_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        next_row[:, layer_index] = token
        embedded = model.model.embedding_list[layer_index](token)
        current_local_input = model.project_global_to_local(embedded)

    return next_row


def _run_local_depth_cached(
    model: MossTTSLocalModel,
    *,
    global_hidden: mx.array,
    sequences: mx.array,
    current_length: int,
    n_vq_for_inference: int,
    config: MossTTSLocalGenerationConfig,
    kv_cache: LocalKVCache,
) -> mx.array:
    batch_size = int(global_hidden.shape[0])
    channels = model.config.channels
    max_depth = min(channels, 1 + n_vq_for_inference)
    kv_cache.reset()
    current_local_input = model.project_global_to_local(global_hidden)
    next_row = mx.full((batch_size, channels), model.config.audio_pad_code, dtype=mx.int32)

    for layer_index in range(max_depth):
        local_output = model.decode_local_step(
            current_local_input[:, None, :],
            kv_cache=kv_cache,
        )
        projected = model.local_to_speech_embedding_mlps[layer_index](
            local_output.last_hidden_state[:, -1, :]
        )
        projected = model.layer_norm_before_lm_heads[layer_index](projected)
        logits = _linear_forward(model.lm_heads[layer_index], projected)
        if layer_index != 0:
            logits[:, model.config.audio_pad_code] = mx.array(
                mx.finfo(logits.dtype).min,
                dtype=logits.dtype,
            )
        temperature, top_k, top_p, repetition_penalty, do_sample = _resolve_sampling_config(
            layer_index,
            config,
        )
        previous_tokens = None
        if repetition_penalty != 1.0:
            previous_tokens = sequences[:, :current_length, layer_index]
        token = sample_next_token(
            logits,
            previous_tokens=previous_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )
        next_row[:, layer_index] = token
        embedded = model.model.embedding_list[layer_index](token)
        current_local_input = model.project_global_to_local(embedded)

    return next_row


def _generate_moss_tts_local_uncached(
    model: MossTTSLocalModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    config: MossTTSLocalGenerationConfig,
) -> MossTTSLocalGenerationOutput:
    config = config or MossTTSLocalGenerationConfig()
    batch_size, prompt_length, channels = input_ids.shape
    n_vq_for_inference = (
        config.n_vq_for_inference
        if config.n_vq_for_inference is not None
        else model.config.n_vq
    )
    n_vq_for_inference = max(1, min(model.config.n_vq, int(n_vq_for_inference)))

    generation_limit = _resolve_generation_limit(config)
    sequences, current_length = _allocate_sequence_buffer(
        input_ids,
        max_new_tokens=generation_limit,
    )
    current_attention_mask, _ = _allocate_attention_mask_buffer(
        attention_mask,
        max_new_tokens=generation_limit,
    )
    stop_reached = False

    for _ in range(generation_limit):
        global_output = model.model(
            input_ids=sequences[:, :current_length, :],
            attention_mask=current_attention_mask[:, :current_length],
            n_vq_for_inference=n_vq_for_inference,
        )
        global_hidden = global_output.last_hidden_state[:, -1, :]

        next_row = _run_local_depth_full(
            model,
            global_hidden=global_hidden,
            sequences=sequences,
            current_length=current_length,
            n_vq_for_inference=n_vq_for_inference,
            config=config,
        )
        sequences[:, current_length, :] = next_row
        current_attention_mask[:, current_length] = mx.ones((batch_size,), dtype=mx.bool_)
        current_length += 1

        if bool(mx.all(next_row[:, 0] == model.config.audio_end_token_id)):
            stop_reached = True
            break

    sequences = sequences[:, :current_length, :]
    generated_rows = sequences[:, prompt_length:current_length, :]
    audio_codes_list = extract_audio_code_sequences(
        sequences,
        prompt_length=prompt_length,
        pad_code=model.config.audio_pad_code,
        n_vq=n_vq_for_inference,
        stop_token=model.config.audio_end_token_id,
    )
    return MossTTSLocalGenerationOutput(
        sequences=sequences,
        generated_rows=generated_rows,
        audio_codes_list=audio_codes_list,
        stop_reached=stop_reached,
    )


def _generate_moss_tts_local_cached(
    model: MossTTSLocalModel,
    input_ids: mx.array,
    attention_mask: mx.array,
    *,
    config: MossTTSLocalGenerationConfig,
) -> MossTTSLocalGenerationOutput:
    batch_size, prompt_length, channels = input_ids.shape
    if batch_size != 1:
        raise ValueError("Cached generation currently supports batch_size == 1 only.")

    n_vq_for_inference = (
        config.n_vq_for_inference
        if config.n_vq_for_inference is not None
        else model.config.n_vq
    )
    n_vq_for_inference = max(1, min(model.config.n_vq, int(n_vq_for_inference)))

    generation_limit = _resolve_generation_limit(config)
    sequences, current_length = _allocate_sequence_buffer(
        input_ids,
        max_new_tokens=generation_limit,
    )
    stop_reached = False

    global_output, global_cache = model.model.prefill(
        input_ids=input_ids,
        attention_mask=attention_mask,
        n_vq_for_inference=n_vq_for_inference,
        max_cache_len=prompt_length + generation_limit,
    )
    global_hidden = global_output.last_hidden_state[:, -1, :]
    local_kv_cache = LocalKVCache.allocate(
        model.local_transformer_config,
        batch_size=batch_size,
        max_length=min(channels, 1 + n_vq_for_inference),
        dtype=global_hidden.dtype,
    )

    for _ in range(generation_limit):
        next_row = _run_local_depth_cached(
            model,
            global_hidden=global_hidden,
            sequences=sequences,
            current_length=current_length,
            n_vq_for_inference=n_vq_for_inference,
            config=config,
            kv_cache=local_kv_cache,
        )
        sequences[:, current_length, :] = next_row
        current_length += 1

        if bool(mx.all(next_row[:, 0] == model.config.audio_end_token_id)):
            stop_reached = True
            break

        global_output = model.model.decode_step(
            input_ids=next_row[:, None, :],
            kv_cache=global_cache,
            n_vq_for_inference=n_vq_for_inference,
        )
        global_hidden = global_output.last_hidden_state[:, -1, :]

    sequences = sequences[:, :current_length, :]
    generated_rows = sequences[:, prompt_length:current_length, :]
    audio_codes_list = extract_audio_code_sequences(
        sequences,
        prompt_length=prompt_length,
        pad_code=model.config.audio_pad_code,
        n_vq=n_vq_for_inference,
        stop_token=model.config.audio_end_token_id,
    )
    return MossTTSLocalGenerationOutput(
        sequences=sequences,
        generated_rows=generated_rows,
        audio_codes_list=audio_codes_list,
        stop_reached=stop_reached,
    )


def _find_last_equal(values: list[int], target: int) -> int:
    for index in range(len(values) - 1, -1, -1):
        if values[index] == target:
            return index
    raise ValueError(f"Target token {target} not found in sequence.")


def _build_decode_outputs(
    sequences: mx.array,
    prompt_lengths: list[int],
    *,
    input_width: int,
    audio_start_token_id: int,
) -> list[tuple[int, mx.array]]:
    outputs: list[tuple[int, mx.array]] = []
    for batch_index, prompt_length in enumerate(prompt_lengths):
        sequence = sequences[batch_index]
        channel_zero = [int(token) for token in sequence[:, 0].tolist()]
        prompt_offset = int(input_width - prompt_length)
        prompt_channel_zero = channel_zero[prompt_offset : prompt_offset + prompt_length]
        relative_start_index = _find_last_equal(prompt_channel_zero, audio_start_token_id)
        start_index = prompt_offset + relative_start_index
        start_length = int(prompt_length - relative_start_index - 1)
        outputs.append((start_length, sequence[start_index:]))
    return outputs


def _merge_audio_segments(audio_segments: tuple[mx.array, ...]) -> mx.array:
    if not audio_segments:
        return mx.zeros((0,), dtype=mx.float32)
    if len(audio_segments) == 1:
        return audio_segments[0].astype(mx.float32)
    return mx.concatenate([segment.astype(mx.float32) for segment in audio_segments], axis=0)


def _decode_generated_audio_codes(
    codec: MossAudioTokenizerModel,
    audio_codes: mx.array,
) -> tuple[mx.array, tuple[mx.array, ...]]:
    if int(audio_codes.shape[0]) == 0:
        empty = mx.zeros((0,), dtype=mx.float32)
        return empty, ()
    codec_input = audio_codes.transpose(1, 0).astype(mx.int32)
    decoded = codec.decode(codec_input, num_quantizers=int(codec_input.shape[0]))
    waveform = decoded.audio[0, 0, : int(decoded.audio_lengths[0])].astype(mx.float32)
    return waveform, (waveform,)


def synthesize_moss_tts_local_conversations(
    model: MossTTSLocalModel,
    processor: MossTTSLocalProcessor,
    codec: MossAudioTokenizerModel,
    *,
    conversations: list[list[dict[str, Any]] | list[AssistantMessage] | list[dict[str, Any] | AssistantMessage]],
    mode: str = "generation",
    config: MossTTSLocalGenerationConfig | None = None,
) -> MossTTSLocalBatchSynthesisOutput:
    config = config or MossTTSLocalGenerationConfig()
    processor.with_audio_tokenizer(codec)
    batch = processor(conversations, mode=mode)
    generation = generate_moss_tts_local(
        model,
        batch.input_ids,
        batch.attention_mask,
        config=config,
    )
    prompt_lengths = [int(length) for length in mx.sum(batch.attention_mask.astype(mx.int32), axis=1).tolist()]
    decode_outputs = _build_decode_outputs(
        generation.sequences,
        prompt_lengths,
        input_width=int(batch.input_ids.shape[1]),
        audio_start_token_id=model.config.audio_start_token_id,
    )
    messages = processor.decode_sequences(decode_outputs)

    outputs: list[MossTTSLocalSynthesisOutput] = []
    for batch_index, message in enumerate(messages):
        waveform, audio_segments = _decode_generated_audio_codes(
            codec,
            generation.audio_codes_list[batch_index],
        )
        content = None
        if message is not None:
            content = message.content
        sample_generation = MossTTSLocalGenerationOutput(
            sequences=generation.sequences[batch_index : batch_index + 1],
            generated_rows=generation.generated_rows[batch_index : batch_index + 1],
            audio_codes_list=(generation.audio_codes_list[batch_index],),
            stop_reached=generation.stop_reached,
        )
        outputs.append(
            MossTTSLocalSynthesisOutput(
                generation=sample_generation,
                waveform=waveform,
                sample_rate=codec.sampling_rate,
                content=content,
                audio_segments=audio_segments,
            )
        )

    return MossTTSLocalBatchSynthesisOutput(
        generation=generation,
        outputs=tuple(outputs),
        sample_rate=codec.sampling_rate,
    )


def synthesize_moss_tts_local(
    model: MossTTSLocalModel,
    processor: MossTTSLocalProcessor,
    codec: MossAudioTokenizerModel,
    *,
    text: str,
    expected_tokens: int | None = None,
    use_duration_estimate: bool = False,
    config: MossTTSLocalGenerationConfig | None = None,
) -> MossTTSLocalSynthesisOutput:
    config = config or MossTTSLocalGenerationConfig.app_defaults()
    resolved_tokens = expected_tokens
    if resolved_tokens is None and use_duration_estimate:
        _, resolved_tokens, _, _ = estimate_duration_tokens(text)

    result = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=[[processor.build_user_message(text=text, tokens=resolved_tokens)]],
        mode="generation",
        config=config,
    )
    return result.outputs[0]
