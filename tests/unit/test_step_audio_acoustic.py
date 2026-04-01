"""Pure Step-Audio acoustic frontend, CAMPPlus, and HiFT tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.step_audio_editx import (
    StepAudioCampPlusConfig,
    StepAudioCampPlusModel,
    StepAudioHiFTConfig,
    StepAudioHiFTF0PredictorConfig,
    StepAudioHiFTGenerator,
    mel_spectrogram,
)
from mlx_speech.models.step_audio_editx.hift import (
    StepAudioSourceModuleHnNSF2,
    _istft,
    _periodic_hann_window,
)


def test_mel_spectrogram_matches_expected_frame_shape() -> None:
    sample_rate = 24000
    audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * np.arange(sample_rate, dtype=np.float32) / sample_rate)

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


def test_campplus_random_weight_forward_shapes() -> None:
    model = StepAudioCampPlusModel(
        StepAudioCampPlusConfig(
            embedding_size=16,
            growth_rate=4,
            init_channels=8,
            block_layers=(1, 1, 1),
            segment_pool_size=10,
        )
    )
    features = np.ones((1, 40, 80), dtype=np.float32)

    embedding = model(mx.array(features, dtype=mx.float32))

    assert embedding.shape == (1, 16)


def _tiny_hift_config() -> StepAudioHiFTConfig:
    return StepAudioHiFTConfig(
        in_channels=80,
        base_channels=16,
        nb_harmonics=2,
        sampling_rate=24000,
        nsf_alpha=0.1,
        nsf_sigma=0.003,
        nsf_voiced_threshold=10.0,
        upsample_rates=(2, 2),
        upsample_kernel_sizes=(4, 4),
        istft_n_fft=8,
        istft_hop_len=2,
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3),),
        source_resblock_kernel_sizes=(3, 3),
        source_resblock_dilation_sizes=((1, 3), (1, 3)),
        lrelu_slope=0.1,
        audio_limit=0.99,
        f0_predictor=StepAudioHiFTF0PredictorConfig(
            num_class=1,
            in_channels=80,
            cond_channels=16,
        ),
    )


def test_hift_random_weight_inference_returns_finite_waveform() -> None:
    model = StepAudioHiFTGenerator(_tiny_hift_config())
    mel = np.zeros((1, 80, 5), dtype=np.float32)

    waveform, source = model.inference(mel)

    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
    assert source.shape[0] == 1
    assert source.shape[1] == 1
    assert np.isfinite(waveform).all()
    assert np.isfinite(source).all()


def test_hift_decode_clips_magnitude_before_istft(monkeypatch: pytest.MonkeyPatch) -> None:
    model = StepAudioHiFTGenerator(_tiny_hift_config())
    freq_bins = model.config.istft_n_fft // 2 + 1
    time_steps = 4
    decoded = np.zeros((1, freq_bins * 2, time_steps), dtype=np.float32)
    decoded[:, :freq_bins, :] = 10.0

    monkeypatch.setattr(model, "_stft", lambda x: (np.zeros((1, freq_bins, time_steps), dtype=np.float32), np.zeros((1, freq_bins, time_steps), dtype=np.float32)))
    monkeypatch.setattr(model, "decode_without_stft", lambda x, s_stft: decoded)

    captured: dict[str, np.ndarray] = {}

    def fake_istft(magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        captured["magnitude"] = magnitude
        captured["phase"] = phase
        return np.zeros((1, time_steps * 32), dtype=np.float32)

    monkeypatch.setattr(model, "_istft", fake_istft)
    waveform = model.decode(
        mx.zeros((1, 80, 2), dtype=mx.float32),
        mx.zeros((1, 1, 32), dtype=mx.float32),
    )

    assert waveform.shape == (1, time_steps * 32)
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
