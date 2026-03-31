import math

import mlx.core as mx

from mlx_voice.generation import MossTTSDelayGenerationConfig, generate_moss_tts_delay
from mlx_voice.generation.moss_delay import (
    _apply_top_k_delay,
    _apply_top_p_delay,
    _build_delay_sampling_audio_mask,
    _initialize_delay_state,
    _resolve_do_sample,
    _sample_delay_token,
    _update_delay_state,
)


class _FakeDelayConfig:
    n_vq = 2
    pad_token_id = 0
    im_start_token_id = 10
    im_end_token_id = 11
    audio_vocab_size = 16
    audio_user_slot_token_id = 12
    audio_assistant_gen_slot_token_id = 13
    audio_assistant_delay_slot_token_id = 14
    audio_start_token_id = 15
    audio_end_token_id = 16
    audio_pad_code = 16
    channels = 3


class _FakeDelayModel:
    def __init__(self):
        self.config = _FakeDelayConfig()
        self.calls = 0

    def __call__(self, *, input_ids, attention_mask, output_hidden_states=False):
        _ = input_ids, attention_mask, output_hidden_states
        self.calls += 1
        batch = 1
        vocab = 32
        audio_vocab = 17
        text_logits = mx.full((batch, 1, vocab), -1e9, dtype=mx.float32)
        audio0_logits = mx.full((batch, 1, audio_vocab), -1e9, dtype=mx.float32)
        audio1_logits = mx.full((batch, 1, audio_vocab), -1e9, dtype=mx.float32)

        if self.calls == 1:
            text_logits[:, :, self.config.audio_start_token_id] = 0
            audio0_logits[:, :, 1] = 0
            audio1_logits[:, :, 2] = 0
        elif self.calls == 2:
            text_logits[:, :, self.config.audio_assistant_delay_slot_token_id] = 0
            audio0_logits[:, :, 3] = 0
            audio1_logits[:, :, self.config.audio_pad_code] = 0
        elif self.calls == 3:
            text_logits[:, :, self.config.audio_assistant_delay_slot_token_id] = 0
            audio0_logits[:, :, self.config.audio_pad_code] = 0
            audio1_logits[:, :, self.config.audio_pad_code] = 0
        elif self.calls == 4:
            text_logits[:, :, self.config.audio_end_token_id] = 0
            audio0_logits[:, :, self.config.audio_pad_code] = 0
            audio1_logits[:, :, self.config.audio_pad_code] = 0
        else:
            text_logits[:, :, self.config.im_end_token_id] = 0
            audio0_logits[:, :, self.config.audio_pad_code] = 0
            audio1_logits[:, :, self.config.audio_pad_code] = 0

        class _Output:
            def __init__(self, logits_all):
                self.logits_all = logits_all

        return _Output((text_logits, audio0_logits, audio1_logits))


class _FakeCachedDelayModel(_FakeDelayModel):
    def __init__(self):
        super().__init__()
        self.prefill_calls = 0
        self.decode_calls = 0

    def prefill(self, *, input_ids, attention_mask, max_cache_len, output_hidden_states=False):
        _ = input_ids, attention_mask, max_cache_len, output_hidden_states
        self.prefill_calls += 1

        class _FakeKVCache:
            current_length = 0

        return self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        ), _FakeKVCache()

    def decode_step(self, *, input_ids, kv_cache, output_hidden_states=False):
        _ = kv_cache
        self.decode_calls += 1
        return self(
            input_ids=input_ids,
            attention_mask=mx.ones((1, 1), dtype=mx.bool_),
            output_hidden_states=output_hidden_states,
        )


def test_delay_generation_loop_terminates_cleanly_with_fake_model() -> None:
    model = _FakeDelayModel()
    input_ids = mx.array([[[10, 16, 16], [20, 16, 16], [21, 16, 16]]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.bool_)

    output = generate_moss_tts_delay(
        model,
        input_ids,
        attention_mask,
        config=MossTTSDelayGenerationConfig(max_new_tokens=8, do_sample=False),
    )

    assert output.stop_reached
    assert output.sequences.shape[1] > input_ids.shape[1]
    assert len(output.messages) == 1
    assert output.messages[0][1].ndim == 2


def test_delay_cached_generation_uses_prefill_and_decode_step() -> None:
    model = _FakeCachedDelayModel()
    input_ids = mx.array([[[10, 16, 16], [20, 16, 16], [21, 16, 16]]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.bool_)

    output = generate_moss_tts_delay(
        model,
        input_ids,
        attention_mask,
        config=MossTTSDelayGenerationConfig(max_new_tokens=8, do_sample=False, use_kv_cache=True),
    )

    assert output.stop_reached
    assert model.prefill_calls == 1
    assert model.decode_calls == int(output.generated_rows.shape[1]) - 1


def test_delay_cached_and_uncached_paths_match_on_deterministic_fake_model() -> None:
    input_ids = mx.array([[[10, 16, 16], [20, 16, 16], [21, 16, 16]]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1]], dtype=mx.bool_)

    uncached_model = _FakeCachedDelayModel()
    uncached = generate_moss_tts_delay(
        uncached_model,
        input_ids,
        attention_mask,
        config=MossTTSDelayGenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            use_kv_cache=False,
        ),
    )

    cached_model = _FakeCachedDelayModel()
    cached = generate_moss_tts_delay(
        cached_model,
        input_ids,
        attention_mask,
        config=MossTTSDelayGenerationConfig(
            max_new_tokens=8,
            do_sample=False,
            use_kv_cache=True,
        ),
    )

    assert uncached.stop_reached == cached.stop_reached
    assert uncached.generated_rows.tolist() == cached.generated_rows.tolist()
    assert uncached.messages[0][1].tolist() == cached.messages[0][1].tolist()


def test_delay_sampling_applies_repetition_penalty_in_greedy_mode() -> None:
    logits = mx.array([[10.0, 9.5, -1.0]], dtype=mx.float32)
    previous = mx.array([[0, 0, 0]], dtype=mx.int32)

    token = _sample_delay_token(
        logits,
        previous_tokens=previous,
        repetition_penalty=2.0,
        top_p=1.0,
        top_k=50,
        do_sample=False,
    )

    assert token.tolist() == [1]


def test_delay_sampling_flattens_prev_tokens_for_selected_audio_rows() -> None:
    logits = mx.array(
        [
            [8.0, 7.5, -1.0],
            [5.0, 4.5, -1.0],
        ],
        dtype=mx.float32,
    )
    previous = mx.array([[[0, 2], [0, 2], [0, 2]]], dtype=mx.int32)

    token = _sample_delay_token(
        logits,
        previous_tokens=previous,
        repetition_penalty=2.0,
        top_p=1.0,
        top_k=50,
        do_sample=False,
    )

    assert token.tolist() == [1, 1]


def test_delay_top_k_keeps_exact_number_of_indices() -> None:
    logits = mx.array([[5.0, 4.0, 4.0, 1.0]], dtype=mx.float32)

    filtered = _apply_top_k_delay(logits, top_k=2)
    filtered_values = filtered[0].tolist()
    finite_indices = [index for index, value in enumerate(filtered_values) if math.isfinite(value)]

    assert len(finite_indices) == 2
    assert 0 in finite_indices
    assert math.isinf(filtered_values[3]) and filtered_values[3] < 0


def test_delay_top_p_masks_removed_logits_to_negative_infinity() -> None:
    logits = mx.array([[4.0, 3.0, 2.0, 1.0]], dtype=mx.float32)

    filtered = _apply_top_p_delay(logits, top_p=0.6)
    filtered_values = filtered[0].tolist()

    assert math.isfinite(filtered_values[0])
    assert any(math.isinf(value) and value < 0 for value in filtered_values[1:])


def test_delay_zero_temperature_forces_greedy_even_with_override() -> None:
    temperature, do_sample = _resolve_do_sample(0.0, True)

    assert temperature == 1.0
    assert do_sample is False


def test_initialize_delay_state_detects_continuation_audio_prefix() -> None:
    input_ids = mx.array(
        [
            [
                [10, 16, 16],
                [15, 16, 16],
                [13, 1, 2],
            ]
        ],
        dtype=mx.int32,
    )

    is_stopping, audio_lengths, delayed_lengths, is_audio = _initialize_delay_state(
        input_ids,
        audio_start_token_id=_FakeDelayConfig.audio_start_token_id,
        audio_assistant_gen_slot_token_id=_FakeDelayConfig.audio_assistant_gen_slot_token_id,
        max_int64=(1 << 63) - 1,
    )

    assert is_stopping.tolist() == [False]
    assert audio_lengths.tolist() == [2]
    assert delayed_lengths.tolist() == [(1 << 63) - 1]
    assert is_audio.tolist() == [True]


def test_build_delay_sampling_audio_mask_respects_audio_and_delay_lengths() -> None:
    sampling_mask = _build_delay_sampling_audio_mask(
        audio_lengths=mx.array([3], dtype=mx.int64),
        delayed_lengths=mx.array([1], dtype=mx.int64),
        is_stopping=mx.array([False], dtype=mx.bool_),
        n_vq=3,
        max_int64=(1 << 63) - 1,
    )

    assert sampling_mask.tolist() == [[False, True, True]]


def test_update_delay_state_transitions_stopping_and_delay_unwind() -> None:
    next_text_tokens = mx.array(
        [
            _FakeDelayConfig.audio_assistant_delay_slot_token_id,
            _FakeDelayConfig.im_end_token_id,
        ],
        dtype=mx.int32,
    )
    audio_lengths = mx.array([0, 2], dtype=mx.int64)
    delayed_lengths = mx.array([(1 << 63) - 1, 1], dtype=mx.int64)
    is_audio = mx.array([False, True], dtype=mx.bool_)
    is_stopping = mx.array([False, False], dtype=mx.bool_)

    (
        next_audio_lengths,
        next_delayed_lengths,
        next_is_audio,
        next_is_stopping,
    ) = _update_delay_state(
        next_text_tokens,
        audio_lengths=audio_lengths,
        delayed_lengths=delayed_lengths,
        is_audio=is_audio,
        is_stopping=is_stopping,
        audio_start_token_id=_FakeDelayConfig.audio_start_token_id,
        audio_end_token_id=_FakeDelayConfig.audio_end_token_id,
        audio_assistant_gen_slot_token_id=_FakeDelayConfig.audio_assistant_gen_slot_token_id,
        audio_assistant_delay_slot_token_id=_FakeDelayConfig.audio_assistant_delay_slot_token_id,
        im_end_token_id=_FakeDelayConfig.im_end_token_id,
        n_vq=_FakeDelayConfig.n_vq,
        max_int64=(1 << 63) - 1,
    )

    assert next_audio_lengths.tolist() == [1, 2]
    assert next_delayed_lengths.tolist() == [0, 2]
    assert next_is_audio.tolist() == [False, True]
    assert next_is_stopping.tolist() == [False, True]
