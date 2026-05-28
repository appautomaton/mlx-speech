"""Multi-modal guider implementing CFG + STG + rescale + modality.

Reference: `.references/DramaBox/ltx2/ltx_core/components/guiders.py:235-305`

The guider is purely arithmetic on the four denoised predictions
``(cond, uncond, ptb, mod)`` — it doesn't run the DiT itself. The caller
(`euler_denoising_loop`) decides which extra passes to make based on the
``enabled_*`` flags and dispatches them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class GuiderParams:
    """Warm-server defaults: ``cfg=2.5, stg=1.5, stg_block=29, rescale="auto",
    modality=1.0``."""

    cfg_scale: float = 2.5
    stg_scale: float = 1.5
    stg_blocks: tuple[int, ...] = (29,)
    rescale_scale: float = 0.0   # 0 disables rescale; "auto" maps to a sigma-aware value
    modality_scale: float = 1.0

    @property
    def needs_uncond(self) -> bool:
        return not math.isclose(self.cfg_scale, 1.0)

    @property
    def needs_ptb(self) -> bool:
        return not math.isclose(self.stg_scale, 0.0)

    @property
    def needs_modality(self) -> bool:
        return not math.isclose(self.modality_scale, 1.0)


class MultiModalGuider:
    """Apply CFG/STG/rescale/modality to a tuple of denoised predictions."""

    def __init__(self, params: GuiderParams):
        self.params = params

    def __call__(
        self,
        cond: mx.array,
        *,
        uncond: mx.array | None = None,
        ptb: mx.array | None = None,
        modality: mx.array | None = None,
    ) -> mx.array:
        """Compute the guided prediction.

        Args:
            cond: conditioned denoised prediction.
            uncond: unconditioned (negative-prompt) prediction; required iff
                ``cfg_scale != 1``.
            ptb: perturbed prediction (e.g. with STG self-attn skip); required
                iff ``stg_scale != 0``.
            modality: isolated-modality prediction; required iff
                ``modality_scale != 1``.
        Returns:
            Guided prediction in the same shape as ``cond``.
        """
        pred = cond
        if self.params.needs_uncond:
            if uncond is None:
                raise ValueError("`uncond` required for non-unit cfg_scale")
            pred = pred + (self.params.cfg_scale - 1.0) * (cond - uncond)
        if self.params.needs_ptb:
            if ptb is None:
                raise ValueError("`ptb` required for non-zero stg_scale")
            pred = pred + self.params.stg_scale * (cond - ptb)
        if self.params.needs_modality:
            if modality is None:
                raise ValueError("`modality` required for non-unit modality_scale")
            pred = pred + (self.params.modality_scale - 1.0) * (cond - modality)

        if self.params.rescale_scale != 0.0:
            cond_std = mx.std(cond.astype(mx.float32))
            pred_std = mx.std(pred.astype(mx.float32)) + 1e-8
            factor = cond_std / pred_std
            factor = self.params.rescale_scale * factor + (1.0 - self.params.rescale_scale)
            pred = pred * factor.astype(pred.dtype)

        return pred


__all__ = ["GuiderParams", "MultiModalGuider"]
