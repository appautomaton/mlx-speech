from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

from mlx_speech.models.dramabox.audio_vae.reference_prep import prepare_reference_audio


TARGET_PEAK = 10.0 ** (-4.0 / 20.0)


def test_prepare_reference_audio_loops_mono_to_stereo_16k(tmp_path):
    sample_rate = 8_000
    samples = np.linspace(-0.25, 0.5, sample_rate // 4, dtype=np.float32)
    path = tmp_path / "mono.wav"
    sf.write(path, samples, sample_rate)

    prepared = prepare_reference_audio(path, ref_duration_s=0.5)

    assert prepared.sample_rate == 16_000
    assert prepared.waveform.shape == (1, 2, 8_000)
    assert mx.allclose(prepared.waveform[:, 0], prepared.waveform[:, 1], atol=1e-6).item()
    peak = float(mx.max(mx.abs(prepared.waveform)).item())
    assert peak == pytest.approx(TARGET_PEAK, abs=0.01)


def test_prepare_reference_audio_crops_long_input(tmp_path):
    sample_rate = 16_000
    left = np.linspace(-0.2, 0.7, sample_rate, dtype=np.float32)
    right = np.linspace(0.4, -0.3, sample_rate, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo, sample_rate)

    prepared = prepare_reference_audio(path, ref_duration_s=0.25)

    assert prepared.waveform.shape == (1, 2, 4_000)
    actual = np.asarray(prepared.waveform)
    assert actual[0, 0, -1] > actual[0, 0, 0]
    assert actual[0, 1, -1] < actual[0, 1, 0]


def test_prepare_reference_audio_rejects_silent_input(tmp_path):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(1600, dtype=np.float32), 16_000)

    with pytest.raises(ValueError, match="silent"):
        prepare_reference_audio(path, ref_duration_s=0.1)
