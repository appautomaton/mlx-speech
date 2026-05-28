"""LTX-2 sigma scheduler.

Generates the ``[steps+1]`` sigma schedule used by the Euler sampler. The
formula:

    sigma_shift = (tokens - BASE) * (max_shift - base_shift) / (MAX - BASE) + base_shift
    sigmas[0..steps] = linspace(1, 0, steps+1)
    sigmas[i] = exp(sigma_shift) / (exp(sigma_shift) + (1/sigma_i - 1)) for sigma_i > 0
    sigmas[i] = 0 otherwise

If ``stretch=True`` (default), the surviving non-zero sigmas are rescaled so
the last one equals ``1 - terminal``.

For DramaBox the patchified latent shape is ``[B, T, 128]`` (the patchifier
collapses ``channels × mel_bins = 8 * 16 = 128``), so
``tokens = math.prod(latent.shape[2:]) = 128`` is essentially constant per
checkpoint regardless of duration. The shift therefore lands well below
``BASE_ANCHOR=1024`` and produces a fairly aggressive early-step schedule.

Reference: `.references/DramaBox/ltx2/ltx_core/components/schedulers.py:14-57`
"""

from __future__ import annotations

import math

import mlx.core as mx

BASE_SHIFT_ANCHOR = 1024
MAX_SHIFT_ANCHOR = 4096


class LTX2Scheduler:
    """Default LTX-2 sigma schedule."""

    def execute(
        self,
        steps: int,
        *,
        tokens: int,
        max_shift: float = 2.05,
        base_shift: float = 0.95,
        stretch: bool = True,
        terminal: float = 0.1,
    ) -> mx.array:
        """Compute the sigma schedule.

        Args:
            steps: number of denoising steps. The output has ``steps + 1``
                values, last being 0.
            tokens: token count (``math.prod(latent.shape[2:])``). For DramaBox
                patchified shape this is 128.
        Returns:
            ``mx.array[float32]`` of length ``steps + 1``.
        """
        sigmas = mx.linspace(1.0, 0.0, steps + 1, dtype=mx.float32)

        mm = (max_shift - base_shift) / (MAX_SHIFT_ANCHOR - BASE_SHIFT_ANCHOR)
        b = base_shift - mm * BASE_SHIFT_ANCHOR
        sigma_shift = tokens * mm + b
        exp_shift = math.exp(sigma_shift)

        # `sigmas[i]` is `exp_shift / (exp_shift + (1/sigma_i - 1))` for non-zero.
        # Handle zero positions via `mx.where` to avoid division by zero.
        # `(1/sigma - 1)` blows up for sigma=0; mask first.
        eps = 1e-12
        nonzero = sigmas != 0.0
        safe = mx.where(nonzero, sigmas, mx.array(eps, dtype=mx.float32))
        ratio = 1.0 / safe - 1.0
        denom = exp_shift + ratio
        shifted = exp_shift / denom
        sigmas = mx.where(nonzero, shifted, mx.array(0.0, dtype=mx.float32))

        if stretch:
            # Rescale non-zero sigmas so the last non-zero entry equals
            # `1 - terminal`.
            one_minus = 1.0 - sigmas  # [steps+1]
            # The last non-zero index is steps - 1 (since sigmas[steps] == 0).
            # `one_minus[-1] == 1 - 0 == 1` is excluded; we want
            # `one_minus[steps - 1]` (the last non-zero).
            anchor = one_minus[steps - 1]
            scale_factor = anchor / (1.0 - terminal)
            stretched = 1.0 - one_minus / scale_factor
            sigmas = mx.where(nonzero, stretched, mx.array(0.0, dtype=mx.float32))

        return sigmas.astype(mx.float32)


__all__ = ["LTX2Scheduler", "BASE_SHIFT_ANCHOR", "MAX_SHIFT_ANCHOR"]
