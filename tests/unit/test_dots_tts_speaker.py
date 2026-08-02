from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.dots_tts.speaker import (
    CAMPPlus,
    CAMPPlusConfig,
    FrozenBatchNorm,
    SpeakerConditioner,
)


def _tiny_encoder() -> CAMPPlus:
    mx.random.seed(23)
    return CAMPPlus(
        CAMPPlusConfig(
            feature_dim=80,
            embedding_size=12,
            growth_rate=2,
            bottleneck_size=2,
            initial_channels=4,
            block_layers=(1, 1, 1),
        )
    )


def test_frozen_batch_norm_uses_running_statistics() -> None:
    norm = FrozenBatchNorm(2)
    norm.weight = mx.array([2.0, 3.0])
    norm.bias = mx.array([0.5, -0.5])
    norm.running_mean = mx.array([1.0, 2.0])
    norm.running_var = mx.array([4.0, 9.0])
    value = norm(mx.array([[[3.0, 5.0]]]))
    mx.eval(value)
    np.testing.assert_allclose(value, [[[2.5, 2.5]]], atol=1e-5)


def test_tiny_campplus_is_deterministic_and_shaped() -> None:
    model = _tiny_encoder()
    mx.random.seed(29)
    features = mx.random.normal((1, 64, 80))
    first = model(features)
    second = model(features)
    mx.eval(first, second)
    assert first.shape == (1, 12)
    np.testing.assert_allclose(first, second, atol=0.0, rtol=0.0)


def test_speaker_conditioner_scales_and_projects() -> None:
    encoder = _tiny_encoder()
    conditioner = SpeakerConditioner(encoder=encoder, conditioning_dim=16)
    time = np.arange(16_000, dtype=np.float32) / 16_000
    audio = 0.1 * np.sin(2 * np.pi * 220.0 * time)
    first = conditioner(audio, sample_rate=16_000, speaker_scale=1.0)
    second = conditioner(audio, sample_rate=16_000, speaker_scale=1.5)
    mx.eval(
        first.embedding,
        first.scaled_embedding,
        first.projected,
        second.scaled_embedding,
    )
    assert first.embedding.shape == (1, 12)
    assert first.projected.shape == (1, 16)
    assert bool(mx.all(mx.isfinite(first.projected)).item())
    np.testing.assert_allclose(first.scaled_embedding, first.embedding)
    np.testing.assert_allclose(second.scaled_embedding, second.embedding * 1.5)


def test_campplus_and_scale_validation() -> None:
    model = _tiny_encoder()
    with pytest.raises(ValueError, match="expects"):
        model(mx.zeros((1, 10, 81)))
    conditioner = SpeakerConditioner(encoder=model, conditioning_dim=16)
    with pytest.raises(ValueError, match="finite"):
        conditioner(np.zeros(16_000), sample_rate=16_000, speaker_scale=float("nan"))
