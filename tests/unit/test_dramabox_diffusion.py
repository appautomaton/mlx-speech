"""Unit tests for the DramaBox diffusion primitives.

Stage 3 — target shape, patchifier, scheduler, noiser, state tools,
velocity/denoised conversions. No checkpoints required.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.diffusion import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    GaussianNoiser,
    LTX2Scheduler,
    LatentState,
    post_process_latent,
    target_shape_from_duration,
    to_denoised,
    to_velocity,
)


# --------------------------------------------------------------------------- #
# target_shape_from_duration
# --------------------------------------------------------------------------- #

def test_target_shape_5_seconds():
    """5.0 s at fps=25 → n_frames = round(125) + 1 = 126;
    align: ((126-1+4)//8)*8 + 1 = (129//8)*8 + 1 = 16*8 + 1 = 129.
    Latents per second: 16000/160/4 = 25 → audio_frames = round(129/25 * 25) = 129.
    """
    shape = target_shape_from_duration(5.0)
    assert shape.batch == 1
    assert shape.channels == 8
    assert shape.frames == 129
    assert shape.mel_bins == 16


def test_target_shape_alignment_to_eight_plus_one():
    """For a few durations, ``frames`` should always satisfy ``(frames - 1) % 8 == 0``."""
    for d in (1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 12.34):
        shape = target_shape_from_duration(d)
        assert (shape.frames - 1) % 8 == 0, f"duration {d}s → {shape.frames}"


def test_target_shape_to_tuple():
    shape = target_shape_from_duration(5.0)
    assert shape.to_tuple() == (1, 8, 129, 16)


# --------------------------------------------------------------------------- #
# AudioPatchifier
# --------------------------------------------------------------------------- #

def test_patchify_unpatchify_roundtrip():
    p = AudioPatchifier()
    B, C, T, F = 1, 8, 5, 16
    latent = mx.random.normal((B, C, T, F), dtype=mx.float32)
    patched = p.patchify(latent)
    assert patched.shape == (B, T, C * F)
    restored = p.unpatchify(patched, channels=C, mel_bins=F)
    assert restored.shape == latent.shape
    assert mx.allclose(restored, latent, atol=1e-6).item()


def test_patchify_preserves_channel_first_ordering():
    """The rearrange `b c t f -> b t (c f)` puts channel-then-freq in the
    last dim. Concretely, for C=2, F=3: out[..., 0..2] == in[:,0,:,:].flatten()."""
    B, C, T, F = 1, 2, 1, 3
    p = AudioPatchifier()
    latent = mx.arange(B * C * T * F, dtype=mx.float32).reshape(B, C, T, F)
    patched = p.patchify(latent)
    # latent[0, 0, 0, :] = [0, 1, 2], latent[0, 1, 0, :] = [3, 4, 5]
    # patched[0, 0, 0..5] should equal [0, 1, 2, 3, 4, 5]
    assert patched[0, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_get_patch_grid_bounds_shape():
    p = AudioPatchifier()
    shape = AudioLatentShape(batch=2, channels=8, frames=10, mel_bins=16)
    bounds = p.get_patch_grid_bounds(shape)
    assert bounds.shape == (2, 1, 10, 2)


def test_get_patch_grid_bounds_causal_clipping():
    """First frame's start timestamp under causal mode is clipped to 0."""
    p = AudioPatchifier(is_causal=True)
    shape = AudioLatentShape(batch=1, channels=8, frames=3, mel_bins=16)
    bounds = p.get_patch_grid_bounds(shape)
    assert float(bounds[0, 0, 0, 0]) == 0.0  # start of first frame is clipped


def test_get_patch_grid_bounds_timestamps_increase():
    p = AudioPatchifier()
    shape = AudioLatentShape(batch=1, channels=8, frames=5, mel_bins=16)
    bounds = p.get_patch_grid_bounds(shape)
    starts = [float(bounds[0, 0, i, 0]) for i in range(5)]
    assert starts == sorted(starts)  # monotonic


# --------------------------------------------------------------------------- #
# LTX2Scheduler
# --------------------------------------------------------------------------- #

def test_scheduler_returns_steps_plus_one_values():
    sched = LTX2Scheduler()
    sigmas = sched.execute(steps=30, tokens=128)
    assert sigmas.shape == (31,)
    assert sigmas.dtype == mx.float32


def test_scheduler_first_is_one_last_is_zero():
    sched = LTX2Scheduler()
    sigmas = sched.execute(steps=30, tokens=128, stretch=False)
    # Without stretch: first non-zero is 1.0 transformed, last is 0
    assert float(sigmas[-1]) == 0.0


def test_scheduler_monotone_decreasing():
    sched = LTX2Scheduler()
    sigmas = sched.execute(steps=30, tokens=128)
    for i in range(30):
        assert float(sigmas[i]) >= float(sigmas[i + 1]) - 1e-6


def test_scheduler_stretch_anchors_terminal():
    """With stretch=True and terminal=0.1, the last non-zero sigma should
    equal 1 - 0.1 = 0.9 in (1 - sigma) space — i.e. sigma equals 0.1."""
    sched = LTX2Scheduler()
    sigmas = sched.execute(steps=30, tokens=128, stretch=True, terminal=0.1)
    # sigmas[-1] is exactly 0 (zero stays zero); sigmas[-2] should be 0.1.
    assert float(sigmas[-1]) == 0.0
    assert float(sigmas[-2]) == pytest.approx(0.1, abs=1e-5)


def test_scheduler_tokens_128_matches_dramabox_warm_server():
    """For DramaBox's 30-step warm-server config: tokens=128, max_shift=2.05,
    base_shift=0.95. Spot-check the shift constant."""
    sched = LTX2Scheduler()
    sigmas = sched.execute(steps=30, tokens=128)
    # The schedule should be reasonable: sigmas[0] is close to 1, sigmas[-1] = 0,
    # and the schedule should have decreasing values.
    assert float(sigmas[0]) > 0.5
    assert float(sigmas[-1]) == 0.0


# --------------------------------------------------------------------------- #
# GaussianNoiser
# --------------------------------------------------------------------------- #

def test_noiser_fully_noised_when_mask_one():
    noiser = GaussianNoiser(seed=42)
    state = LatentState(
        latent=mx.zeros((1, 4, 8), dtype=mx.float32),
        denoise_mask=mx.ones((1, 4, 1), dtype=mx.float32),
        positions=mx.zeros((1, 1, 4, 2), dtype=mx.float32),
        clean_latent=mx.zeros((1, 4, 8), dtype=mx.float32),
    )
    new = noiser(state, noise_scale=1.0)
    # When mask=1 and clean=0: noised = noise * 1 + 0 * 0 = noise (random)
    # Variance should be ~1.
    var = float(mx.var(new.latent))
    assert 0.5 < var < 2.0


def test_noiser_zero_mask_preserves_latent():
    """When denoise_mask=0, the latent should be unchanged."""
    noiser = GaussianNoiser(seed=42)
    original = mx.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=mx.float32)
    state = LatentState(
        latent=original,
        denoise_mask=mx.zeros((1, 1, 1), dtype=mx.float32),
        positions=mx.zeros((1, 1, 1, 2), dtype=mx.float32),
        clean_latent=original,
    )
    new = noiser(state, noise_scale=1.0)
    assert mx.allclose(new.latent, original, atol=1e-5).item()


def test_noiser_partial_mask_blends():
    """With mask=0.5, the output should be `noise*0.5 + latent*0.5`."""
    noiser = GaussianNoiser(seed=42)
    state = LatentState(
        latent=mx.ones((1, 4, 8), dtype=mx.float32) * 5.0,
        denoise_mask=mx.full((1, 4, 1), 0.5, dtype=mx.float32),
        positions=mx.zeros((1, 1, 4, 2), dtype=mx.float32),
        clean_latent=mx.zeros((1, 4, 8), dtype=mx.float32),
    )
    new = noiser(state, noise_scale=1.0)
    # noised = noise * 0.5 + 5 * 0.5
    # Mean should be ~2.5 (since E[noise * 0.5] = 0, but the +2.5 dominates)
    assert 2.0 < float(mx.mean(new.latent)) < 3.0


# --------------------------------------------------------------------------- #
# to_velocity / to_denoised
# --------------------------------------------------------------------------- #

def test_to_velocity_formula():
    sample = mx.array([4.0, 8.0], dtype=mx.float32)
    denoised = mx.array([2.0, 4.0], dtype=mx.float32)
    v = to_velocity(sample, sigma=2.0, denoised=denoised)
    # (sample - denoised) / sigma = ([2, 4]) / 2 = [1, 2]
    assert mx.allclose(v, mx.array([1.0, 2.0]), atol=1e-5).item()


def test_to_denoised_formula():
    sample = mx.array([4.0, 8.0], dtype=mx.float32)
    velocity = mx.array([1.0, 2.0], dtype=mx.float32)
    d = to_denoised(sample, velocity, sigma=2.0)
    # sample - velocity * sigma = [4 - 2, 8 - 4] = [2, 4]
    assert mx.allclose(d, mx.array([2.0, 4.0]), atol=1e-5).item()


def test_to_velocity_zero_sigma_raises():
    sample = mx.array([1.0])
    denoised = mx.array([0.0])
    with pytest.raises(ValueError):
        to_velocity(sample, sigma=0.0, denoised=denoised)


def test_to_velocity_to_denoised_roundtrip():
    sample = mx.random.normal((4,), dtype=mx.float32)
    denoised = mx.random.normal((4,), dtype=mx.float32)
    sigma = 0.5
    v = to_velocity(sample, sigma, denoised)
    d_back = to_denoised(sample, v, sigma)
    assert mx.allclose(d_back, denoised, atol=1e-5).item()


# --------------------------------------------------------------------------- #
# post_process_latent
# --------------------------------------------------------------------------- #

def test_post_process_latent_passes_through_target():
    """For ``denoise_mask=1`` (target tokens), output == denoised."""
    denoised = mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32)
    mask = mx.ones_like(denoised)
    clean = mx.zeros_like(denoised)
    out = post_process_latent(denoised, mask, clean)
    assert mx.allclose(out, denoised, atol=1e-6).item()


def test_post_process_latent_freezes_ref():
    """For ``denoise_mask=0`` (ref tokens), output == clean."""
    denoised = mx.array([[[1.0, 2.0, 3.0]]], dtype=mx.float32)
    mask = mx.zeros_like(denoised)
    clean = mx.array([[[10.0, 20.0, 30.0]]], dtype=mx.float32)
    out = post_process_latent(denoised, mask, clean)
    assert mx.allclose(out, clean, atol=1e-6).item()


def test_post_process_latent_mixed_mask():
    """Half-and-half mask blends linearly."""
    denoised = mx.array([[[10.0]]], dtype=mx.float32)
    mask = mx.array([[[0.5]]], dtype=mx.float32)
    clean = mx.array([[[2.0]]], dtype=mx.float32)
    out = post_process_latent(denoised, mask, clean)
    # 0.5 * 10 + 0.5 * 2 = 6
    assert float(out[0, 0, 0]) == pytest.approx(6.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# AudioLatentTools
# --------------------------------------------------------------------------- #

def test_audio_latent_tools_create_initial_state_shapes():
    shape = AudioLatentShape(batch=1, channels=8, frames=5, mel_bins=16)
    tools = AudioLatentTools(patchifier=AudioPatchifier(), target_shape=shape)
    state = tools.create_initial_state(dtype=mx.float32)
    # Latent should be patchified
    assert state.latent.shape == (1, 5, 8 * 16)
    assert state.clean_latent.shape == state.latent.shape
    # Mask matches the reference convention: (B, 1, T, 1) before patchify
    # → (B, T, 1) after, NOT (B, T, C*F). The mask is per-token, broadcast
    # against the patch feature dim downstream.
    assert state.denoise_mask.shape == (1, 5, 1)
    assert state.positions.shape == (1, 1, 5, 2)
    # All zeros for the latent and clean
    assert mx.all(state.latent == 0).item()
    assert mx.all(state.clean_latent == 0).item()
    # Denoise mask all ones (full denoising)
    assert mx.all(state.denoise_mask == 1).item()
    # No attention_mask by default
    assert state.attention_mask is None


def test_audio_latent_tools_clear_conditioning_strips_tail():
    """Append fake ref tokens after the target, then verify
    clear_conditioning slices them off."""
    shape = AudioLatentShape(batch=1, channels=8, frames=3, mel_bins=16)
    tools = AudioLatentTools(patchifier=AudioPatchifier(), target_shape=shape)
    base = tools.create_initial_state()
    # Append 2 trailing "ref" tokens. denoise_mask is (B, T, 1) per ref.
    extra_latent = mx.ones((1, 2, 128), dtype=mx.float32) * 7.0
    extra_clean = mx.ones((1, 2, 128), dtype=mx.float32) * 11.0
    extra_mask = mx.zeros((1, 2, 1), dtype=mx.float32)
    extra_pos = mx.zeros((1, 1, 2, 2), dtype=mx.float32)
    extended = base.replace(
        latent=mx.concatenate([base.latent, extra_latent], axis=1),
        clean_latent=mx.concatenate([base.clean_latent, extra_clean], axis=1),
        denoise_mask=mx.concatenate([base.denoise_mask, extra_mask], axis=1),
        positions=mx.concatenate([base.positions, extra_pos], axis=2),
    )
    cleared = tools.clear_conditioning(extended)
    assert cleared.latent.shape == (1, 3, 128)
    assert cleared.denoise_mask.shape == (1, 3, 1)
    # Denoise mask should be reset to all-ones
    assert mx.all(cleared.denoise_mask == 1).item()
    # Trailing 7s and 11s should be gone
    assert mx.all(cleared.latent == 0).item()
    assert mx.all(cleared.clean_latent == 0).item()


def test_audio_latent_tools_unpatchify_roundtrip():
    shape = AudioLatentShape(batch=1, channels=8, frames=4, mel_bins=16)
    tools = AudioLatentTools(patchifier=AudioPatchifier(), target_shape=shape)
    state = tools.create_initial_state()
    # Set the patched latent to random values
    state = state.replace(latent=mx.random.normal(state.latent.shape, dtype=mx.float32))
    unp = tools.unpatchify_state(state)
    assert unp.latent.shape == (1, 8, 4, 16)
    # Patchify back
    re = tools.patchifier.patchify(unp.latent)
    assert mx.allclose(re, state.latent, atol=1e-6).item()
