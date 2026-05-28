"""Auto-rescale schedule for CFG.

The warm-server CLI exposes ``rescale="auto"`` which maps the static cfg
scale to a per-step latent std-rescale strength. This is a verbatim port of
`auto_rescale_for_cfg` from `.references/DramaBox/src/inference_server.py:91-116`.

The CFG formula ``pred = cond + (cfg-1)*(cond - uncond)`` makes ``pred.std()``
grow with cfg, which the VAE+vocoder render as progressively louder (and
eventually clipping) waveforms. The rescale strength pulls ``pred.std()`` back
toward ``cond.std()`` and is tuned per cfg to keep ≥1 dB peak headroom while
preserving the extra punch of high-cfg generations.
"""

from __future__ import annotations


def auto_rescale_for_cfg(cfg_scale: float) -> float:
    """CFG-aware std-rescale strength. Verbatim port of the upstream schedule.

    Reference values (`inference_server.py:91-116`):
        cfg ≤ 2.0  → 0.0   (no rescale; CFG barely applied)
        cfg = 2.5  → 0.30
        cfg = 3.0  → 0.60
        cfg = 4.0  → 0.80
        cfg ∈ [4, 8] → 0.80  (plateau — keep high-cfg punch)
        cfg = 10.0 → 1.00
    """
    if cfg_scale <= 2.0:
        return 0.0
    if cfg_scale <= 3.0:
        return 0.6 * (cfg_scale - 2.0)               # 0.0 → 0.6
    if cfg_scale <= 4.0:
        return 0.6 + 0.2 * (cfg_scale - 3.0)         # 0.6 → 0.8
    if cfg_scale <= 8.0:
        return 0.8                                    # plateau
    return min(1.0, 0.8 + 0.1 * (cfg_scale - 8.0))   # 0.8 → 1.0 at cfg=10


__all__ = ["auto_rescale_for_cfg"]
