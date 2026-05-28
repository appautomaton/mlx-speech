"""Gaussian noiser — convex mix of noise and current latent.

Reference: `.references/DramaBox/ltx2/ltx_core/components/noisers.py:15-35`

The convex mix is:

    scaled_mask = denoise_mask * noise_scale
    noised = noise * scaled_mask + latent * (1 - scaled_mask)

For full-noise initialization (clean latent = 0, denoise_mask = 1,
noise_scale = 1), this reduces to ``noised = noise``. For IC-LoRA voice-ref
conditioning the reference tokens have ``denoise_mask = 0`` so they remain
unchanged.
"""

from __future__ import annotations

import mlx.core as mx

from .state import LatentState


class GaussianNoiser:
    """Sample Gaussian noise from a fixed key and blend with the latent."""

    def __init__(self, seed: int):
        self._key = mx.random.key(int(seed))

    def __call__(self, state: LatentState, *, noise_scale: float = 1.0) -> LatentState:
        """Return a new state with ``latent`` blended toward Gaussian noise."""
        # Sample noise in fp32 then cast to the latent's dtype.
        noise = mx.random.normal(state.latent.shape, key=self._key, dtype=mx.float32)
        noise = noise.astype(state.latent.dtype)

        scaled_mask = state.denoise_mask * noise_scale
        # Broadcast scaled_mask to the latent's shape if needed.
        if scaled_mask.shape != state.latent.shape:
            scaled_mask = mx.broadcast_to(scaled_mask, state.latent.shape)
        scaled_mask = scaled_mask.astype(state.latent.dtype)

        noised = noise * scaled_mask + state.latent * (1.0 - scaled_mask)
        return state.replace(latent=noised)


__all__ = ["GaussianNoiser"]
