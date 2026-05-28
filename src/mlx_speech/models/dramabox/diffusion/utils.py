"""Velocity / denoised utilities and the post-step re-blend.

Reference:
- `.references/DramaBox/ltx2/ltx_core/utils.py:21-52` — `to_velocity`, `to_denoised`
- `.references/DramaBox/ltx2/ltx_pipelines/utils/helpers.py:252-254` — `post_process_latent`
"""

from __future__ import annotations

import mlx.core as mx


def to_velocity(sample: mx.array, sigma: float | mx.array, denoised: mx.array) -> mx.array:
    """``(sample - denoised) / sigma`` in fp32, cast back to ``sample.dtype``.

    Raises if ``sigma == 0``.
    """
    sigma_val = float(sigma) if isinstance(sigma, mx.array) else float(sigma)
    if sigma_val == 0.0:
        raise ValueError("to_velocity: sigma must be non-zero")
    out = (sample.astype(mx.float32) - denoised.astype(mx.float32)) / sigma_val
    return out.astype(sample.dtype)


def to_denoised(sample: mx.array, velocity: mx.array, sigma: float | mx.array) -> mx.array:
    """``sample - velocity * sigma`` in fp32, cast back to ``sample.dtype``.

    Used to wrap the DiT's velocity output into the ``x0`` (denoised)
    prediction the sampler expects.
    """
    sigma_val = float(sigma) if isinstance(sigma, mx.array) else float(sigma)
    out = sample.astype(mx.float32) - velocity.astype(mx.float32) * sigma_val
    return out.astype(sample.dtype)


def post_process_latent(
    denoised: mx.array,
    denoise_mask: mx.array,
    clean_latent: mx.array,
) -> mx.array:
    """Re-blend frozen reference tokens before each Euler step.

    Formula::

        denoised = mask * denoised + (1 - mask) * clean

    Where ``mask`` is the denoise_mask (1 for target tokens that should be
    denoised, 0 for frozen reference tokens). This is applied BEFORE the
    Euler step (see `samplers.py:20-31` in the upstream code), not after.
    """
    return denoise_mask * denoised + (1.0 - denoise_mask) * clean_latent


__all__ = ["to_velocity", "to_denoised", "post_process_latent"]
