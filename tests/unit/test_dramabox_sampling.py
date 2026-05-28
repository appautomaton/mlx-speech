"""Unit tests for the DramaBox sampling primitives (Stage 7)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.sampling import (
    GuiderParams,
    MultiModalGuider,
    auto_rescale_for_cfg,
    silence_prior_fix,
)


# --------------------------------------------------------------------------- #
# auto_rescale_for_cfg
# --------------------------------------------------------------------------- #

def test_auto_rescale_cfg_unit_returns_zero():
    assert auto_rescale_for_cfg(1.0) == 0.0


def test_auto_rescale_matches_reference_schedule():
    """Verbatim port of inference_server.py:91-116 — check the anchor points."""
    expected = {
        1.0: 0.0,
        2.0: 0.0,
        2.5: 0.30,
        3.0: 0.60,
        4.0: 0.80,
        7.0: 0.80,
        8.0: 0.80,
        10.0: 1.00,
    }
    for cfg, want in expected.items():
        assert auto_rescale_for_cfg(cfg) == pytest.approx(want), f"cfg={cfg}"


def test_auto_rescale_cfg_25_matches_reference():
    assert auto_rescale_for_cfg(2.5) == pytest.approx(0.30)


def test_auto_rescale_monotone():
    """rescale should be non-decreasing in cfg."""
    vals = [auto_rescale_for_cfg(c) for c in (1.0, 1.2, 1.5, 2.0, 2.5, 4.0, 10.0)]
    for a, b in zip(vals, vals[1:]):
        assert a <= b + 1e-6


# --------------------------------------------------------------------------- #
# MultiModalGuider
# --------------------------------------------------------------------------- #

def test_guider_unit_cfg_passes_cond_through():
    """cfg=1, stg=0, rescale=0, mod=1 → guided pred == cond."""
    g = MultiModalGuider(GuiderParams(cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0, modality_scale=1.0))
    cond = mx.random.normal((1, 4, 8), dtype=mx.float32)
    pred = g(cond)
    assert mx.allclose(pred, cond, atol=1e-6).item()


def test_guider_cfg_only():
    """cfg=2.0: pred = cond + 1 * (cond - uncond) = 2*cond - uncond."""
    g = MultiModalGuider(GuiderParams(cfg_scale=2.0, stg_scale=0.0, rescale_scale=0.0, modality_scale=1.0))
    cond = mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32)
    uncond = mx.array([[[0.5, 1.0, 1.5]]], dtype=mx.float32)
    pred = g(cond, uncond=uncond)
    expected = 2 * cond - uncond
    assert mx.allclose(pred, expected, atol=1e-6).item()


def test_guider_stg_only():
    """stg=1.0: pred = cond + 1*(cond - ptb) = 2*cond - ptb."""
    g = MultiModalGuider(GuiderParams(cfg_scale=1.0, stg_scale=1.0, rescale_scale=0.0))
    cond = mx.array([[[1.0, 2.0]]], dtype=mx.float32)
    ptb = mx.array([[[0.5, 1.0]]], dtype=mx.float32)
    pred = g(cond, ptb=ptb)
    expected = 2 * cond - ptb
    assert mx.allclose(pred, expected, atol=1e-6).item()


def test_guider_missing_uncond_raises():
    g = MultiModalGuider(GuiderParams(cfg_scale=2.0))
    cond = mx.zeros((1, 4, 8))
    with pytest.raises(ValueError):
        g(cond)


def test_guider_rescale_scale_one_matches_std():
    """At rescale_scale=1.0: factor = cond.std/pred.std → pred * (cond.std/pred.std)
    has std == cond.std exactly."""
    g = MultiModalGuider(GuiderParams(cfg_scale=2.0, stg_scale=0.0, rescale_scale=1.0))
    cond = mx.random.normal((1, 16, 32), dtype=mx.float32)
    uncond = mx.random.normal((1, 16, 32), dtype=mx.float32)
    pred = g(cond, uncond=uncond)
    cond_std = float(mx.std(cond))
    pred_std = float(mx.std(pred))
    assert pred_std == pytest.approx(cond_std, rel=1e-3)


# --------------------------------------------------------------------------- #
# silence_prior_fix
# --------------------------------------------------------------------------- #

def test_silence_prior_no_op_short():
    """For T <= 513, the fix is a no-op."""
    latent = mx.random.normal((1, 8, 100, 16), dtype=mx.float32)
    out = silence_prior_fix(latent)
    assert mx.allclose(out, latent, atol=0.0).item()


def test_silence_prior_modifies_frames_512_513():
    """For T = 600, frames 512 and 513 should be linear interpolation of 511 and 514."""
    T = 600
    latent = mx.random.normal((1, 8, T, 16), dtype=mx.float32)
    out = silence_prior_fix(latent)
    # Frames 0..511 unchanged
    assert mx.allclose(out[:, :, :512, :], latent[:, :, :512, :], atol=0.0).item()
    # Frames 514..end unchanged
    assert mx.allclose(out[:, :, 514:, :], latent[:, :, 514:, :], atol=0.0).item()
    # Frame 512 = 2/3 * latent[511] + 1/3 * latent[514]
    expected_512 = latent[:, :, 511, :] * (2 / 3) + latent[:, :, 514, :] * (1 / 3)
    assert mx.allclose(out[:, :, 512, :], expected_512, atol=1e-5).item()
    # Frame 513 = 1/3 * latent[511] + 2/3 * latent[514]
    expected_513 = latent[:, :, 511, :] * (1 / 3) + latent[:, :, 514, :] * (2 / 3)
    assert mx.allclose(out[:, :, 513, :], expected_513, atol=1e-5).item()


# --------------------------------------------------------------------------- #
# GuiderParams flags
# --------------------------------------------------------------------------- #

def test_guider_params_default_flags():
    p = GuiderParams()  # warm-server defaults
    assert p.cfg_scale == 2.5
    assert p.stg_scale == 1.5
    assert p.stg_blocks == (29,)
    assert p.modality_scale == 1.0
    assert p.needs_uncond is True
    assert p.needs_ptb is True
    assert p.needs_modality is False


def test_guider_params_no_guidance_flags():
    p = GuiderParams(cfg_scale=1.0, stg_scale=0.0, modality_scale=1.0)
    assert p.needs_uncond is False
    assert p.needs_ptb is False
    assert p.needs_modality is False
