from __future__ import annotations

import numpy as np
import pytest

from mlx_speech.models.dots_tts.speaker import SpeakerFrontend


def _synthetic_audio(seconds: float = 0.64) -> np.ndarray:
    sample_rate = 48_000
    count = round(seconds * sample_rate)
    time = np.arange(count, dtype=np.float32) / sample_rate
    envelope = np.linspace(0.35, 1.0, count, dtype=np.float32)
    return envelope * (
        0.16 * np.sin(2 * np.pi * 220.0 * time)
        + 0.04 * np.sin(2 * np.pi * 440.0 * time + 0.3)
    )


def test_speaker_fbank_matches_official_oracle() -> None:
    fixture = np.load("tests/fixtures/dots_tts/soar/speaker.npz")
    features, length = SpeakerFrontend().features(
        _synthetic_audio(), sample_rate=48_000
    )
    assert length == int(fixture["fbank_length"][0])
    np.testing.assert_allclose(
        features, fixture["fbank"][0], atol=0.01, rtol=0.01
    )


def test_speaker_frontend_mixes_channels_and_caps_ten_seconds() -> None:
    audio = _synthetic_audio(10.5)
    stereo = np.stack((audio, audio * 0.5), axis=1)
    frontend = SpeakerFrontend(max_audio_seconds=10.0)
    features, length = frontend.features(stereo, sample_rate=48_000)
    assert features.shape == (length, 80)
    assert length == 998


def test_speaker_frontend_rejects_bad_audio() -> None:
    with pytest.raises(ValueError, match="shorter"):
        SpeakerFrontend().features(np.zeros(100), sample_rate=16_000)
    bad = np.zeros(1_000, dtype=np.float32)
    bad[10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        SpeakerFrontend().features(bad, sample_rate=16_000)
