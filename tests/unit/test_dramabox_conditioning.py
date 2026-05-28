from __future__ import annotations

import mlx.core as mx

from mlx_speech.models.dramabox.diffusion import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    apply_reference_latent,
)


def _target_state(frames: int = 4):
    patchifier = AudioPatchifier()
    shape = AudioLatentShape(batch=1, channels=8, frames=frames, mel_bins=16)
    tools = AudioLatentTools(patchifier=patchifier, target_shape=shape)
    return patchifier, tools, tools.create_initial_state(dtype=mx.float32)


def test_apply_reference_latent_appends_tokens_mask_and_positions():
    patchifier, _, state = _target_state(frames=4)
    ref_latent = mx.ones((1, 8, 3, 16), dtype=mx.float32)

    conditioned = apply_reference_latent(state, ref_latent, patchifier=patchifier)

    assert conditioned.latent.shape == (1, 7, 128)
    assert conditioned.clean_latent.shape == (1, 7, 128)
    assert conditioned.denoise_mask.shape == (1, 7, 1)
    assert conditioned.positions.shape == (1, 1, 7, 2)
    assert mx.allclose(conditioned.denoise_mask[:, :4], mx.ones((1, 4, 1)), atol=0.0).item()
    assert mx.allclose(conditioned.denoise_mask[:, 4:], mx.zeros((1, 3, 1)), atol=0.0).item()

    expected_ref_tokens = patchifier.patchify(ref_latent)
    assert mx.allclose(conditioned.latent[:, 4:], expected_ref_tokens, atol=0.0).item()
    assert mx.allclose(conditioned.clean_latent[:, 4:], expected_ref_tokens, atol=0.0).item()

    ref_shape = AudioLatentShape(batch=1, channels=8, frames=3, mel_bins=16)
    expected_ref_positions = patchifier.get_patch_grid_bounds(ref_shape) + 0.5
    assert mx.allclose(conditioned.positions[:, :, :4], state.positions, atol=0.0).item()
    assert mx.allclose(conditioned.positions[:, :, 4:], expected_ref_positions, atol=1e-6).item()


def test_apply_reference_latent_builds_asymmetric_attention_mask():
    patchifier, _, state = _target_state(frames=2)
    ref_latent = mx.ones((1, 8, 2, 16), dtype=mx.float32)

    conditioned = apply_reference_latent(state, ref_latent, patchifier=patchifier)
    mask = conditioned.attention_mask

    assert mask is not None
    assert mask.shape == (1, 1, 4, 4)
    assert mx.allclose(mask[:, :, :2, :], mx.zeros((1, 1, 2, 4)), atol=0.0).item()
    assert mx.allclose(mask[:, :, 2:, 2:], mx.zeros((1, 1, 2, 2)), atol=0.0).item()
    assert float(mx.max(mask[:, :, 2:, :2]).item()) < -1.0e30


def test_clear_conditioning_restores_target_only_state():
    patchifier, tools, state = _target_state(frames=4)
    ref_latent = mx.ones((1, 8, 3, 16), dtype=mx.float32)
    conditioned = apply_reference_latent(state, ref_latent, patchifier=patchifier)

    cleared = tools.clear_conditioning(conditioned)

    assert cleared.latent.shape == state.latent.shape
    assert cleared.clean_latent.shape == state.clean_latent.shape
    assert cleared.positions.shape == state.positions.shape
    assert cleared.attention_mask is None
    assert mx.allclose(cleared.latent, state.latent, atol=0.0).item()
    assert mx.allclose(cleared.clean_latent, state.clean_latent, atol=0.0).item()
    assert mx.allclose(cleared.positions, state.positions, atol=0.0).item()
    assert mx.allclose(cleared.denoise_mask, mx.ones_like(state.denoise_mask), atol=0.0).item()
