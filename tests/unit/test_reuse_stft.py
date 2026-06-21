"""Unit tests for the RE-USE STFT front end (pure-MLX port).

Validates the round-trip ``iSTFT(STFT(x)) ~ x`` (including the relu_log1p
compress/expand), the config-rate param scaling, the sweep-artifact filter
against the reference behavior, and the chunked Hann overlap-add helper.

Reference: `.references/RE-USE/models/stfts.py`,
`.references/DramaBox/src/super_resolution.py:224-265`.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.reuse.stft import (
    chunked_hann_ola,
    hann_window,
    mag_phase_istft,
    mag_phase_stft,
    stft_params_for,
    sweep_artifact_filter,
)


def _interior_max_err(a: np.ndarray, b: np.ndarray, guard: int) -> float:
    """Max abs error ignoring the ``guard`` boundary samples on each side.

    Centered STFT/iSTFT tapers the first/last frames because the COLA
    window-sum is incomplete at the signal edges — torch behaves the same
    way — so we compare the interior where reconstruction is exact.
    """
    length = min(a.shape[-1], b.shape[-1])
    return float(np.max(np.abs(a[guard : length - guard] - b[guard : length - guard])))


@pytest.mark.parametrize("op_sr", [8000, 16000])
def test_stft_param_scaling(op_sr):
    """n_fft/hop/win scale from the 8 kHz config and stay even."""
    n_fft, hop, win = stft_params_for(op_sr)
    factor = op_sr // 8000
    assert (n_fft, hop, win) == (320 * factor, 40 * factor, 320 * factor)
    assert n_fft % 2 == 0 and hop % 2 == 0 and win % 2 == 0


@pytest.mark.parametrize("op_sr", [8000, 16000])
def test_round_trip_relu_log1p(op_sr):
    """iSTFT(STFT(x)) reconstructs x through the relu_log1p compression."""
    n_fft, hop, win = stft_params_for(op_sr)
    rng = np.random.default_rng(0)
    sig = (rng.standard_normal(op_sr) * 0.3).astype(np.float32)
    wave = mx.array(sig)[None, :]

    mag, pha = mag_phase_stft(wave, n_fft, hop, win, "relu_log1p")
    assert mag.shape == pha.shape
    assert mag.shape[1] == n_fft // 2 + 1

    rec = mag_phase_istft(mag, pha, n_fft, hop, win, "relu_log1p")
    rec_np = np.asarray(rec)[0]
    err = _interior_max_err(sig, rec_np, guard=n_fft)
    assert err < 1e-3, f"round-trip error {err:.2e} too large at op_sr={op_sr}"


def test_round_trip_realish_sine():
    """Round-trip on a multi-tone (realistic) signal holds tightly."""
    op_sr = 16000
    n_fft, hop, win = stft_params_for(op_sr)
    t = np.arange(op_sr) / op_sr
    sig = (
        0.4 * np.sin(2 * np.pi * 220 * t)
        + 0.2 * np.sin(2 * np.pi * 440 * t)
        + 0.1 * np.sin(2 * np.pi * 1500 * t)
    ).astype(np.float32)
    wave = mx.array(sig)[None, :]

    mag, pha = mag_phase_stft(wave, n_fft, hop, win, "relu_log1p")
    rec = np.asarray(mag_phase_istft(mag, pha, n_fft, hop, win, "relu_log1p"))[0]
    err = _interior_max_err(sig, rec, guard=n_fft)
    assert err < 1e-3, f"sine round-trip error {err:.2e} too large"


def test_round_trip_power_compress():
    """A numeric power compress_factor round-trips (forward/inverse symmetry)."""
    op_sr = 8000
    n_fft, hop, win = stft_params_for(op_sr)
    rng = np.random.default_rng(7)
    sig = (rng.standard_normal(op_sr) * 0.25).astype(np.float32)
    wave = mx.array(sig)[None, :]

    mag, pha = mag_phase_stft(wave, n_fft, hop, win, 0.3)
    rec = np.asarray(mag_phase_istft(mag, pha, n_fft, hop, win, 0.3))[0]
    err = _interior_max_err(sig, rec, guard=n_fft)
    assert err < 1e-3, f"power-compress round-trip error {err:.2e}"


def test_stft_accepts_1d_and_2d():
    """1-D input round-trips to 1-D; 2-D input round-trips to 2-D."""
    n_fft, hop, win = stft_params_for(8000)
    rng = np.random.default_rng(3)
    sig = (rng.standard_normal(4000) * 0.2).astype(np.float32)

    mag1, pha1 = mag_phase_stft(mx.array(sig), n_fft, hop, win, "relu_log1p")
    assert mag1.ndim == 2
    rec1 = mag_phase_istft(mag1, pha1, n_fft, hop, win, "relu_log1p")
    assert rec1.ndim == 1

    mag2, pha2 = mag_phase_stft(mx.array(sig)[None, :], n_fft, hop, win, "relu_log1p")
    assert mag2.ndim == 3
    rec2 = mag_phase_istft(mag2, pha2, n_fft, hop, win, "relu_log1p")
    assert rec2.ndim == 2


def test_hann_window_periodic():
    """hann_window matches the periodic torch.hann_window formula."""
    win = 320
    n = np.arange(win)
    expected = 0.5 - 0.5 * np.cos(2 * np.pi * n / win)
    got = np.asarray(hann_window(win))
    assert np.allclose(got, expected, atol=1e-6)


def test_sweep_artifact_filter_reference_behavior():
    """A >50%-zero frame is zeroed; 50%-zero and normal frames are kept.

    Mirrors `super_resolution.py:251-254`: mag = expm1(relu(amp)); a frame is
    zeroed when its zero-bin fraction exceeds 0.5. relu(amp<=0) -> 0 ->
    expm1(0) = 0, so non-positive amplitudes count as zero bins.
    """
    # (B=1, n_freq=4, n_frames=3)
    #   frame 0: all positive          -> 0/4 zero        -> kept
    #   frame 1: three non-positive     -> 3/4 = 0.75 > .5 -> zeroed
    #   frame 2: two non-positive       -> 2/4 = 0.50      -> kept
    amp = mx.array(
        [
            [
                [1.0, -5.0, -5.0],
                [1.0, -5.0, -5.0],
                [1.0, -5.0, 1.0],
                [1.0, 1.0, 1.0],
            ]
        ]
    )
    out = np.asarray(sweep_artifact_filter(amp))[0]

    assert np.allclose(out[:, 0], np.asarray(amp)[0][:, 0]), "normal frame altered"
    assert np.allclose(out[:, 1], 0.0), ">50%-zero frame not zeroed"
    assert np.allclose(out[:, 2], np.asarray(amp)[0][:, 2]), "50%-zero frame altered"


def test_sweep_artifact_filter_accepts_2d():
    """2-D amplitude (n_freq, n_frames) is handled and returns 2-D."""
    amp = mx.array([[1.0, -5.0], [1.0, -5.0], [1.0, -5.0], [1.0, 1.0]])
    out = sweep_artifact_filter(amp)
    assert out.ndim == 2
    out_np = np.asarray(out)
    assert np.allclose(out_np[:, 1], 0.0)  # 3/4 zero -> zeroed


def test_chunked_hann_ola_identity():
    """Identity processing reconstructs the input through Hann OLA."""
    rng = np.random.default_rng(1)
    sig = (rng.standard_normal(5000) * 0.2).astype(np.float32)
    wave = mx.array(sig)

    rec = chunked_hann_ola(lambda chunk: chunk, wave, chunk_size=1000)
    rec_np = np.asarray(rec)
    assert rec_np.shape == sig.shape
    err = _interior_max_err(sig, rec_np, guard=500)
    assert err < 1e-4, f"chunked OLA identity error {err:.2e}"


def test_chunked_hann_ola_short_signal():
    """A signal shorter than one chunk is handled as a single chunk."""
    rng = np.random.default_rng(5)
    sig = (rng.standard_normal(300) * 0.2).astype(np.float32)
    wave = mx.array(sig)[None, :]

    rec = chunked_hann_ola(lambda chunk: chunk, wave, chunk_size=1000)
    assert rec.shape == (1, 300)


def test_chunked_hann_ola_pads_short_output():
    """A process that returns a shorter chunk is padded back to chunk length."""
    rng = np.random.default_rng(9)
    sig = (rng.standard_normal(3000) * 0.2).astype(np.float32)
    wave = mx.array(sig)

    # Drop the last 5 samples of each chunk; OLA must still produce full length.
    rec = chunked_hann_ola(lambda chunk: chunk[:, :-5], wave, chunk_size=1000)
    assert rec.shape == sig.shape
