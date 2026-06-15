"""Runtime self-consistency test for the RE-USE / SEMamba enhancer (Slice 5).

Tier-3 runtime test: loads the converted MLX SEMamba weights and runs the full
chunked `enhance()` pipeline (STFT -> SEMamba -> sweep filter -> iSTFT -> Hann
overlap-add) on a synthetic noisy waveform.

This is NOT numeric parity against the torch reference (that is Slice 6's
fixture gate). The assertions are deliberately robust: the output is finite,
length-matched, amplitude-bounded, differs from the input (the model did
something), and its energy is not blown up. A strict SNR-improvement assertion
on a synthetic non-speech signal would be flaky (RE-USE is a speech model), so
it is intentionally avoided.

Skips cleanly when the converted MLX weights are absent (CI without weights);
runs for real locally where `models/reuse/mlx/` is present.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

MLX_WEIGHTS = Path("models/reuse/mlx")

pytestmark = pytest.mark.skipif(
    not (MLX_WEIGHTS / "model.safetensors").is_file(),
    reason="converted RE-USE MLX weights not present (run scripts/convert/reuse.py)",
)

_SR = 16_000
_DURATION_S = 2.5


def _noisy_signal(sr: int, duration_s: float) -> mx.array:
    """A low-amplitude tone plus white noise, bounded well inside [-1, 1]."""
    rng = np.random.default_rng(0)
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr
    tone = 0.3 * np.sin(2.0 * np.pi * 220.0 * t).astype(np.float32)
    noise = 0.1 * rng.standard_normal(n).astype(np.float32)
    return mx.array(tone + noise)


def test_enhance_is_finite_length_matched_and_changes_signal():
    from mlx_speech.generation.reuse import REUSEEnhancer

    enhancer = REUSEEnhancer.from_dir(MLX_WEIGHTS)

    noisy = _noisy_signal(_SR, _DURATION_S)
    clean = enhancer.enhance(noisy, _SR)
    mx.eval(clean)

    # Mono, same length as the input.
    assert clean.ndim == 1
    assert clean.shape[0] == noisy.shape[0]

    clean_np = np.array(clean)
    noisy_np = np.array(noisy)

    # Finite and amplitude-bounded (clamped to [-1, 1] inside enhance).
    assert np.all(np.isfinite(clean_np))
    assert np.max(np.abs(clean_np)) <= 1.0 + 1e-6

    # The model did something: the output is not the input.
    assert not np.allclose(clean_np, noisy_np, atol=1e-4)

    # Energy is not blown up (gentle bound, not a denoising SNR claim).
    in_rms = float(np.sqrt(np.mean(noisy_np**2)))
    out_rms = float(np.sqrt(np.mean(clean_np**2)))
    assert out_rms <= 4.0 * in_rms + 1e-6


def test_enhance_accepts_2d_mono_input():
    """A ``(1, T)`` input returns a ``(1, T)`` output of the same length."""
    from mlx_speech.generation.reuse import REUSEEnhancer

    enhancer = REUSEEnhancer.from_dir(MLX_WEIGHTS)

    noisy = _noisy_signal(_SR, _DURATION_S)[None, :]
    clean = enhancer.enhance(noisy, _SR)
    mx.eval(clean)

    assert clean.ndim == 2
    assert clean.shape == noisy.shape
    assert np.all(np.isfinite(np.array(clean)))
