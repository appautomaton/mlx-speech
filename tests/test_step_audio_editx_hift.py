"""Tests for the Step-Audio HiFT vocoder slice."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.models.step_audio_editx import (
    StepAudioCosyVoiceFrontEnd,
    load_step_audio_flow_conditioner,
    load_step_audio_flow_model,
    load_step_audio_hift_model,
)
from mlx_speech.models.step_audio_editx.hift import (
    StepAudioSourceModuleHnNSF2,
    _istft,
    _periodic_hann_window,
)


MODEL_DIR = Path("models/stepfun/step_audio_editx/original")
COSYVOICE_DIR = MODEL_DIR / "CosyVoice-300M-25Hz"
HAS_ASSETS = COSYVOICE_DIR.exists()


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_hift_checkpoint_alignment_is_exact() -> None:
    loaded = load_step_audio_hift_model(MODEL_DIR)

    assert loaded.alignment_report.is_exact_match
    assert loaded.config.sampling_rate == 24000
    assert loaded.config.upsample_rates == (8, 5, 3)
    assert loaded.config.istft_n_fft == 16
    assert loaded.config.istft_hop_len == 4


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_hift_inference_returns_waveform_shape_for_synthetic_mel() -> None:
    loaded = load_step_audio_hift_model(MODEL_DIR)
    mel = np.zeros((1, 80, 10), dtype=np.float32)

    waveform, source = loaded.model.inference(mel)

    assert waveform.shape == (1, 4800)
    assert source.shape == (1, 1, 4800)
    assert waveform.dtype == np.float32
    assert source.dtype == np.float32
    assert np.isfinite(waveform).all()
    assert np.isfinite(source).all()


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_flow_mel_output_decodes_to_waveform_via_hift() -> None:
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(MODEL_DIR)
    conditioner = load_step_audio_flow_conditioner(MODEL_DIR)
    flow_model = load_step_audio_flow_model(MODEL_DIR)
    hift_model = load_step_audio_hift_model(MODEL_DIR)

    sample_rate = 24000
    audio = 0.1 * np.sin(
        2.0 * np.pi * 330.0 * np.arange(sample_rate, dtype=np.float32) / sample_rate
    )
    prompt_feat, _ = frontend.extract_speech_feat(audio, sample_rate)
    speaker_embedding = frontend.extract_spk_embedding(audio, sample_rate)
    prepared = conditioner.model.prepare_nonstream_inputs(
        token=[20, 21, 1027, 1028, 1029, 30, 31],
        prompt_token=[0, 1, 1024, 1025, 1026],
        prompt_feat=prompt_feat,
        speaker_embedding=speaker_embedding,
    )

    mel = flow_model.model.inference(prepared, n_timesteps=2)
    waveform, source = hift_model.model.inference(mel)

    assert mel.shape == (1, 80, 12)
    assert waveform.shape == (1, 5760)
    assert source.shape == (1, 1, 5760)
    assert np.isfinite(waveform).all()
    assert np.isfinite(source).all()


@pytest.mark.skipif(not HAS_ASSETS, reason="CosyVoice assets not available")
def test_hift_decode_clips_magnitude_before_istft(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = load_step_audio_hift_model(MODEL_DIR)
    model = loaded.model
    freq_bins = loaded.config.istft_n_fft // 2 + 1
    time_steps = 4
    decoded = np.zeros((1, freq_bins * 2, time_steps), dtype=np.float32)
    decoded[:, :freq_bins, :] = 10.0  # exp(10) would exceed the upstream 1e2 clamp

    monkeypatch.setattr(model, "_stft", lambda x: (np.zeros((1, freq_bins, time_steps), dtype=np.float32), np.zeros((1, freq_bins, time_steps), dtype=np.float32)))
    monkeypatch.setattr(model, "decode_without_stft", lambda x, s_stft: decoded)

    captured: dict[str, np.ndarray] = {}

    def fake_istft(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        captured["magnitude"] = magnitude
        captured["phase"] = phase
        return np.zeros((1, time_steps * 480), dtype=np.float32)

    monkeypatch.setattr(model, "_istft", fake_istft)

    waveform = model.decode(
        np.zeros((1, 80, 2), dtype=np.float32),
        np.zeros((1, 1, 32), dtype=np.float32),
    )

    assert waveform.shape == (1, time_steps * 480)
    assert "magnitude" in captured
    assert float(np.max(captured["magnitude"])) == pytest.approx(1e2)
    assert np.isfinite(captured["phase"]).all()


def test_sinegen_linear_resize_uses_align_corners_false_semantics() -> None:
    module = StepAudioSourceModuleHnNSF2(
        sampling_rate=24000,
        upsample_scale=120,
        harmonic_num=7,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshold=10.0,
    )
    values = np.array([[[0.0], [10.0], [20.0], [30.0]]], dtype=np.float32)

    resized = module._linear_resize_time(values, 3)

    expected = np.array([[[1.6666666], [15.0], [28.333334]]], dtype=np.float32)
    assert resized.shape == (1, 3, 1)
    assert np.allclose(resized, expected, atol=1e-6, rtol=1e-6)


def test_hift_istft_matches_centered_length_contract() -> None:
    n_fft = 16
    hop_len = 4
    frames = 24
    freq_bins = n_fft // 2 + 1
    magnitude = np.ones((1, freq_bins, frames), dtype=np.float32)
    phase = np.zeros((1, freq_bins, frames), dtype=np.float32)

    waveform = _istft(
        magnitude,
        phase,
        n_fft=n_fft,
        hop_len=hop_len,
        window=_periodic_hann_window(n_fft),
    )

    assert waveform.shape == (1, 92)
    assert np.isfinite(waveform).all()
