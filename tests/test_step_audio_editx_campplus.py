"""Tests for the Step-Audio CAMPPlus speaker-embedding path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.models.step_audio_editx import (
    StepAudioCosyVoiceFrontEnd,
    load_step_audio_campplus_model,
)


MODEL_DIR = Path("models/stepfun/step_audio_editx/original")
COSYVOICE_DIR = MODEL_DIR / "CosyVoice-300M-25Hz"
HAS_ASSETS = COSYVOICE_DIR.exists()


def _sine_wave(sample_rate: int, frequency_hz: float, seconds: float = 1.0) -> np.ndarray:
    sample_count = int(sample_rate * seconds)
    time = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
    return 0.1 * np.sin(2.0 * np.pi * frequency_hz * time)


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_campplus_checkpoint_alignment_is_exact() -> None:
    loaded = load_step_audio_campplus_model(MODEL_DIR)

    assert loaded.alignment_report.is_exact_match
    assert loaded.config.embedding_size == 192
    assert loaded.config.block_layers == (12, 24, 16)


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_campplus_runtime_extract_embedding_is_deterministic() -> None:
    loaded = load_step_audio_campplus_model(MODEL_DIR)
    audio = _sine_wave(sample_rate=16000, frequency_hz=440.0)

    first = loaded.runtime.extract_embedding(audio, 16000)
    second = loaded.runtime.extract_embedding(audio, 16000)

    assert first.shape == (1, 192)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert np.allclose(first, second)


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_frontend_extract_spk_embedding_matches_loaded_runtime() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(MODEL_DIR)
    direct = load_step_audio_campplus_model(MODEL_DIR)
    audio = _sine_wave(sample_rate=24000, frequency_hz=330.0)

    from_frontend = frontend.extract_spk_embedding(audio, 24000)
    from_runtime = direct.runtime.extract_embedding(audio, 24000)

    assert from_frontend.shape == (1, 192)
    assert from_frontend.dtype == np.float32
    assert np.allclose(from_frontend, from_runtime, atol=1e-5, rtol=1e-5)
