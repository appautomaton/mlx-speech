from __future__ import annotations

import mlx.core as mx
import numpy as np

from mlx_speech.models.dots_tts.speaker import (
    CAMPPlus,
    CAMPPlusConfig,
    SpeakerFrontend,
)
from mlx_speech.models.fireredtts3.speaker import FireRedSpeakerEncoder


def test_fire_red_speaker_encoder_returns_finite_embedding() -> None:
    mx.random.seed(19)
    model = CAMPPlus(
        CAMPPlusConfig(
            feature_dim=80,
            embedding_size=12,
            growth_rate=2,
            bottleneck_size=2,
            initial_channels=4,
            block_layers=(1, 1, 1),
        )
    )
    encoder = FireRedSpeakerEncoder(model, max_audio_seconds=2.0)
    time = np.arange(16_000, dtype=np.float32) / 16_000
    audio = 0.1 * np.sin(2 * np.pi * 220.0 * time)
    embedding = encoder(audio, sample_rate=16_000)
    mx.eval(embedding)
    assert embedding.shape == (1, 12)
    assert bool(mx.all(mx.isfinite(embedding)).item())


def test_fire_red_speaker_frontend_matches_official_oracle() -> None:
    fixture = np.load("tests/fixtures/dots_tts/soar/speaker.npz")
    sample_rate = 48_000
    count = round(0.64 * sample_rate)
    time = np.arange(count, dtype=np.float32) / sample_rate
    envelope = np.linspace(0.35, 1.0, count, dtype=np.float32)
    audio = envelope * (
        0.16 * np.sin(2 * np.pi * 220.0 * time)
        + 0.04 * np.sin(2 * np.pi * 440.0 * time + 0.3)
    )
    features, length = SpeakerFrontend().features(audio, sample_rate=sample_rate)
    assert length == int(fixture["fbank_length"][0])
    np.testing.assert_allclose(features, fixture["fbank"][0], atol=0.01, rtol=0.01)
