"""Tests for the Step-Audio CosyVoice frontend slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.models.step_audio_editx import (
    StepAudioCosyVoiceFrontEnd,
    StepAudioCosyVoiceMelConfig,
    mel_spectrogram,
    resolve_step_audio_cosyvoice_dir,
)


MODEL_DIR = Path("models/stepfun/step_audio_editx/original")
COSYVOICE_DIR = MODEL_DIR / "CosyVoice-300M-25Hz"
HAS_ASSETS = COSYVOICE_DIR.exists()


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_cosyvoice_config_parses_local_yaml() -> None:
    config = StepAudioCosyVoiceMelConfig.from_yaml_path(COSYVOICE_DIR / "cosyvoice.yaml")

    assert config.num_mels == 80
    assert config.n_fft == 1920
    assert config.hop_size == 480
    assert config.win_size == 1920
    assert config.sampling_rate == 24000
    assert config.fmin == 0.0
    assert config.fmax == 8000.0


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_resolve_cosyvoice_dir_from_step_audio_root() -> None:
    assert resolve_step_audio_cosyvoice_dir(MODEL_DIR) == COSYVOICE_DIR
    assert resolve_step_audio_cosyvoice_dir(COSYVOICE_DIR) == COSYVOICE_DIR


def test_mel_spectrogram_matches_expected_frame_shape() -> None:
    seconds = 1
    sample_rate = 24000
    audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate)

    mel = mel_spectrogram(
        audio,
        n_fft=1920,
        num_mels=80,
        sampling_rate=24000,
        hop_size=480,
        win_size=1920,
        fmin=0.0,
        fmax=8000.0,
    )

    assert mel.shape == (80, 50)
    assert np.isfinite(mel).all()


def test_mel_spectrogram_honors_fmax_cutoff_for_high_frequency_tone() -> None:
    sample_rate = 24000
    tone_hz = 10000.0
    audio = 0.1 * np.sin(
        2.0 * np.pi * tone_hz * np.arange(sample_rate, dtype=np.float32) / sample_rate
    )

    mel_cutoff = mel_spectrogram(
        audio,
        n_fft=1920,
        num_mels=80,
        sampling_rate=sample_rate,
        hop_size=480,
        win_size=1920,
        fmin=0.0,
        fmax=8000.0,
    )
    mel_fullband = mel_spectrogram(
        audio,
        n_fft=1920,
        num_mels=80,
        sampling_rate=sample_rate,
        hop_size=480,
        win_size=1920,
        fmin=0.0,
        fmax=12000.0,
    )

    assert float(mel_fullband.max() - mel_cutoff.max()) > 2.0
    assert float(mel_cutoff.max()) < -3.0


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_extract_speech_feat_resamples_and_returns_prompt_shape() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(MODEL_DIR)
    sample_rate = 16000
    audio = 0.1 * np.sin(2.0 * np.pi * 330.0 * np.arange(sample_rate, dtype=np.float32) / sample_rate)

    speech_feat, speech_feat_len = frontend.extract_speech_feat(audio, sample_rate)

    assert speech_feat.shape == (1, 50, 80)
    assert speech_feat_len.tolist() == [50]
    assert speech_feat.dtype == np.float32
