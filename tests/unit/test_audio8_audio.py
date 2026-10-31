from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_speech.models.audio8_asr_infinite.audio_encoder import Audio8Encoder
from mlx_speech.models.audio8_asr_infinite.config import Audio8Config
from mlx_speech.models.audio8_asr_infinite.feature_extraction import (
    Audio8FeatureConfig,
    Audio8FeatureExtractor,
)
from tests.unit.test_audio8_checkpoint import _config_payload


def _audio_config():
    return Audio8Config.from_dict(_config_payload()).audio_config


def test_log_mel_has_voxtral_shape_and_fixed_silence_floor():
    extractor = Audio8FeatureExtractor(Audio8FeatureConfig(feature_size=4))

    features = extractor(np.zeros(1600, dtype=np.float32))

    assert features.shape == (1, 4, 10)
    np.testing.assert_allclose(np.asarray(features), -0.625, rtol=0, atol=1e-6)


def test_embedder_uses_stride_one_then_stride_two():
    encoder = Audio8Encoder(_audio_config())
    for convolution in (encoder.embedder.conv1, encoder.embedder.conv2):
        convolution.weight = convolution.weight.astype(mx.bfloat16)
        convolution.bias = convolution.bias.astype(mx.bfloat16)
    features = mx.ones((1, 4, 10), dtype=mx.float32)

    embedded = encoder.embedder(features)

    assert embedded.shape == (1, 5, 8)
    assert embedded.dtype == mx.bfloat16
    assert encoder.embedder.conv1.left_pad == 2
    assert encoder.embedder.conv2.left_pad == 1
    assert "bias" not in encoder.layers[0].self_attn.k_proj


def test_incremental_audio_encoder_matches_full_sequence_across_window_boundary():
    mx.random.seed(7)
    encoder = Audio8Encoder(_audio_config())
    inputs = mx.random.normal((1, 11, 8))

    reference = encoder(inputs_embeds=inputs).last_hidden_state
    cache = encoder.new_cache()
    chunks = []
    offset = 0
    for size in (3, 1, 4, 3):
        chunks.append(
            encoder(
                inputs_embeds=inputs[:, offset : offset + size], cache=cache
            ).last_hidden_state
        )
        offset += size
    streamed = mx.concatenate(chunks, axis=1)
    mx.eval(reference, streamed)

    np.testing.assert_allclose(
        np.asarray(streamed), np.asarray(reference), rtol=2e-3, atol=1e-3
    )
    assert cache.position_offset == inputs.shape[1]
    assert all(layer.keys.shape[2] == 11 for layer in cache.layers)
    np.testing.assert_array_equal(
        np.asarray(cache.layers[0].positions), np.arange(11)
    )
