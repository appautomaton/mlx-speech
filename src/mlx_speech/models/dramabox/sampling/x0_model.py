"""X0Model wrapper — turns the DiT's velocity output into denoised ``x0``.

The upstream `X0Model(velocity_model)` does:

    velocity = velocity_model(...)
    denoised = to_denoised(latent, velocity, timesteps)

where ``timesteps = sigma * denoise_mask`` (per-token if denoise_mask is
non-uniform). For our v5 baseline path (no voice ref), ``denoise_mask`` is
all-ones, so ``timesteps == sigma`` and the formula collapses to
``denoised = latent - velocity * sigma``.

Reference: `.references/DramaBox/ltx2/ltx_core/model/transformer/model.py:461-486`
"""

from __future__ import annotations

import mlx.core as mx

from ..diffusion.utils import to_denoised
from ..dit.model import LTXModel


class X0Model:
    """Wrap a `LTXModel` (velocity predictor) into an x0 (denoised) predictor."""

    def __init__(self, velocity_model: LTXModel):
        self.velocity_model = velocity_model

    def __call__(
        self,
        latent: mx.array,
        *,
        a_ctx: mx.array,
        sigma: mx.array,
        positions: mx.array | None = None,
        rope_cos_sin: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        """Return denoised ``x0 [B, T, 128]`` for the given inputs.

        Args:
            latent: ``[B, T, 128]`` patchified noisy latent.
            a_ctx: ``[B, T_text, 2048]`` prompt encoder output.
            sigma: per-batch ``[B]`` current diffusion sigma.
            positions: optional ``[B, 1, T, 2]`` patchifier start/end timings;
                passed through to the velocity model for RoPE.
            rope_cos_sin: optional pre-computed RoPE table.
        """
        velocity = self.velocity_model(
            latent,
            a_ctx=a_ctx,
            sigma=sigma,
            positions=positions,
            rope_cos_sin=rope_cos_sin,
        )
        # For our broadcast-sigma baseline (no voice ref), timesteps == sigma scalar.
        return to_denoised(latent, velocity, float(sigma[0]))


__all__ = ["X0Model"]
