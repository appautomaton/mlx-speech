"""DramaBox sampling loop — X0Model wrapper, guidance, Euler denoising loop.

Implements the warm-server denoising flow:

    for sigma_i, sigma_next in zip(sigmas[:-1], sigmas[1:]):
        cond_x0 = X0Model(latent, sigma_i, a_ctx)
        uncond_x0 = X0Model(latent, sigma_i, a_ctx_neg)         if cfg > 1
        ptb_x0   = X0Model(latent, sigma_i, a_ctx, skip_block)  if stg > 0
        pred = cond_x0 + (cfg-1)*(cond-uncond) + stg*(cond-ptb) + (mod-1)*(cond-mod)
        if rescale != 0:
            factor = cond.std/pred.std
            factor = rescale*factor + (1-rescale)
            pred *= factor
        pred = post_process_latent(pred, denoise_mask, clean_latent)
        latent = euler_step(latent, pred, sigma_i, sigma_next)
        mx.eval(latent)
"""

from __future__ import annotations

from .auto_rescale import auto_rescale_for_cfg
from .guider import GuiderParams, MultiModalGuider
from .loop import euler_denoising_loop
from .silence_prior import silence_prior_fix
from .x0_model import X0Model

__all__ = [
    "GuiderParams",
    "MultiModalGuider",
    "X0Model",
    "auto_rescale_for_cfg",
    "euler_denoising_loop",
    "silence_prior_fix",
]
