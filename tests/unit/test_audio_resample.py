"""resample_audio: a windowed-sinc resampler (torchaudio-equivalent), not linear interpolation."""

import math

import mlx.core as mx
import numpy as np

from mlx_speech.audio import resample_audio


def test_resample_matches_torchaudio_on_a_fixed_signal() -> None:
    # torchaudio.functional.resample(x, 24000, 16000) on numpy's default_rng(0) normal draws * 0.5 (240 samples);
    # the first / last values were recorded from torchaudio 2.14 and hold to 3e-7.
    x = np.random.default_rng(0).standard_normal(240).astype(np.float32) * 0.5
    y = np.asarray(resample_audio(mx.array(x), orig_sample_rate=24000, target_sample_rate=16000))

    assert y.shape == (160,)
    expected_head = [-0.0283638, 0.1641691, 0.0588616, -0.1351997, 0.687421, 0.0533735,
                     -0.62998, -0.1345499, -0.7060183, -0.4090299, -0.4203756, -0.2313279]
    expected_tail = [-0.4781788, 0.7957916, -0.3780592, -0.7366494, -0.8155971, -1.1179947]
    assert np.allclose(y[:12], expected_head, atol=2e-6)
    assert np.allclose(y[-6:], expected_tail, atol=2e-6)


def test_resample_rejects_a_tone_above_the_target_nyquist() -> None:
    # a 9 kHz tone at 48 kHz lies above 16 kHz's Nyquist: an anti-aliased resampler attenuates it (torchaudio
    # leaves an rms of 0.10 from the transition band), linear interpolation folds it to 7 kHz at full amplitude (0.71)
    t = np.arange(4800) / 48000.0
    tone = np.sin(2 * math.pi * 9000 * t).astype(np.float32)
    y = np.asarray(resample_audio(mx.array(tone), orig_sample_rate=48000, target_sample_rate=16000))

    assert y.shape == (1600,)
    assert float(np.sqrt((y ** 2).mean())) < 0.15


def test_resample_identity_and_empty() -> None:
    x = mx.array([0.1, -0.2, 0.3], dtype=mx.float32)
    assert np.array_equal(np.asarray(resample_audio(x, orig_sample_rate=16000, target_sample_rate=16000)), np.asarray(x))
    assert resample_audio(mx.zeros((0,), dtype=mx.float32), orig_sample_rate=24000, target_sample_rate=16000).shape == (0,)
