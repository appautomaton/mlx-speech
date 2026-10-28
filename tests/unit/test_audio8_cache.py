from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_speech.models.audio8_asr_infinite.audio_encoder import Audio8RotaryEmbedding
from mlx_speech.models.audio8_asr_infinite.cache import (
    apply_rope_delta,
    trim_text_cache,
)
from mlx_speech.models.audio8_asr_infinite.text_decoder import text_cache_length
from tests.unit.test_audio8_checkpoint import _config_payload
from tests.unit.test_audio8_streaming import _model
from mlx_speech.models.audio8_asr_infinite.config import Audio8Config


def test_rope_negative_delta_matches_direct_rebased_positions():
    config = Audio8Config.from_dict(_config_payload()).audio_config
    rotary = Audio8RotaryEmbedding(config)
    keys = mx.random.normal((1, 2, 3, 4))
    positions = mx.array([3, 7, 11], dtype=mx.int32)

    rotated = rotary.apply(keys, keys, positions)[1]
    rebased = apply_rope_delta(rotated, rotary.inv_freq, -3)
    expected = rotary.apply(keys, keys, positions - 3)[1]
    mx.eval(rebased, expected)

    np.testing.assert_allclose(
        np.asarray(rebased), np.asarray(expected), rtol=2e-5, atol=2e-6
    )


def test_text_trim_preserves_prefix_values_and_rotates_only_tail_keys():
    model = _model()
    inputs = mx.random.normal((1, 21, 8))
    t_cond = mx.zeros((1, 1, 8))
    output = model.language_model(
        inputs_embeds=inputs, t_cond=t_cond, logits_to_keep=1
    )
    original_keys, original_values = output.cache[0].fetch()
    capacity = output.cache[0].capacity

    trim_count = trim_text_cache(
        output.cache,
        stable_prefix=16,
        max_tokens=19,
        inv_freq=model.language_model.model.rotary_emb._inv_freq(),
    )
    keys, values = output.cache[0].fetch()
    expected_tail = apply_rope_delta(
        original_keys[:, 18:], model.language_model.model.rotary_emb._inv_freq(), -2
    )
    mx.eval(keys, values, expected_tail)

    assert trim_count == 2
    assert text_cache_length(output.cache) == 19
    assert output.cache[0].capacity == capacity
    np.testing.assert_array_equal(np.asarray(keys[:, :16]), np.asarray(original_keys[:, :16]))
    np.testing.assert_allclose(np.asarray(keys[:, 16:]), np.asarray(expected_tail), rtol=2e-5, atol=2e-6)
    np.testing.assert_array_equal(
        np.asarray(values),
        np.asarray(mx.concatenate([original_values[:, :16], original_values[:, 18:]], axis=1)),
    )


def test_long_stream_keeps_text_audio_and_pcm_state_bounded():
    model = _model()
    session = model.stream_session(
        language="zh",
        frame_len=2,
        target_delay_ms=80,
        rolling_context_tokens=20,
        rolling_trim_interval_ms=40,
    )
    waveform = np.sin(np.arange(19200, dtype=np.float32) * 0.013)

    session.feed(waveform)
    session.finalize()

    assert len(session.generated_token_ids) > 21
    assert text_cache_length(session.text_cache) <= 20
    assert session.audio_cache.position_offset < 32
    ring_capacity = model.config.audio_config.sliding_window + 2 - 1  # + frame_len - 1
    for layer in session.audio_cache.layers:
        assert layer.keys.shape[2] <= ring_capacity
        assert int(mx.min(layer.positions).item()) >= 0
        assert int(mx.max(layer.positions).item()) < 32
    assert session._pcm.shape[0] < 5000
    assert session.result().language == "zh"
