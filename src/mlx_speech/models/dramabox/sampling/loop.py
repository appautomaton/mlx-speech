"""Euler denoising loop.

For DramaBox the loop runs 30 steps with the warm-server defaults
``(cfg=2.5, stg=1.5, rescale='auto', modality=1.0)``. Each step requires up
to 3 DiT forward passes:

- ``cond``: positive prompt context.
- ``uncond``: negative prompt context (run iff ``cfg_scale != 1``).
- ``ptb``: positive prompt context with the STG self-attn passthrough on
  ``params.stg_blocks`` (run iff ``stg_scale != 0``). The perturbed pass
  mirrors ``cond`` exactly (same sigma, positions, denoise_mask) except for
  the perturbed blocks, matching upstream `denoisers._guided_denoise`.

Reference: `.references/DramaBox/ltx2/ltx_pipelines/utils/denoisers.py:92-174`
"""

from __future__ import annotations

import mlx.core as mx

from ..diffusion.state import LatentState
from ..diffusion.utils import post_process_latent, to_velocity
from .guider import GuiderParams, MultiModalGuider
from .x0_model import X0Model


def euler_denoising_loop(
    state: LatentState,
    sigmas: mx.array,
    *,
    x0_model: X0Model,
    a_ctx: mx.array,
    a_ctx_neg: mx.array | None,
    params: GuiderParams,
    positions: mx.array | None = None,
    rope_cos_sin: tuple[mx.array, mx.array] | None = None,
    denoise_mask: mx.array | None = None,
) -> LatentState:
    """Run the 30-step Euler denoising loop.

    Args:
        state: initial `LatentState` (patchified, noised).
        sigmas: ``[steps + 1]`` schedule (`sigmas[-1] == 0`).
        x0_model: `X0Model` wrapping the DiT velocity predictor.
        a_ctx: prompt encoder output ``[B, T_text, 2048]``.
        a_ctx_neg: negative-prompt ``a_ctx``; required iff `params.needs_uncond`.
        params: `GuiderParams` with cfg/stg/rescale/modality settings.
        positions: optional ``[B, 1, T, 2]`` patchifier start/end timings —
            forwarded to the DiT so RoPE matches the reference.
        rope_cos_sin: optional pre-computed RoPE table; takes precedence over
            ``positions``.
        denoise_mask: optional ``[B, T, 1]`` per-token mask forwarded to the
            x0_model/DiT so reference tokens get per-token timestep 0. Pass
            ``None`` (no voice ref) to keep the bit-identical scalar-sigma path.
    Returns:
        Updated `LatentState` after the final Euler step (`sigma_next == 0`).
    """
    guider = MultiModalGuider(params)
    n_steps = sigmas.shape[0] - 1

    for i in range(n_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        sigma_batched = mx.broadcast_to(sigma[None], (state.latent.shape[0],))

        cond = x0_model(
            state.latent, a_ctx=a_ctx, sigma=sigma_batched,
            positions=positions, rope_cos_sin=rope_cos_sin,
            attention_mask=state.attention_mask,
            denoise_mask=denoise_mask,
        )
        uncond = (
            x0_model(
                state.latent, a_ctx=a_ctx_neg, sigma=sigma_batched,
                positions=positions, rope_cos_sin=rope_cos_sin,
                attention_mask=state.attention_mask,
                denoise_mask=denoise_mask,
            )
            if params.needs_uncond and a_ctx_neg is not None
            else None
        )
        # STG perturbed pass: same positive `a_ctx`, sigma, and denoise_mask as
        # `cond`, but with the self-attn passthrough on `params.stg_blocks`.
        ptb = (
            x0_model(
                state.latent, a_ctx=a_ctx, sigma=sigma_batched,
                positions=positions, rope_cos_sin=rope_cos_sin,
                attention_mask=state.attention_mask,
                denoise_mask=denoise_mask,
                stg_blocks=params.stg_blocks,
            )
            if params.needs_ptb
            else None
        )
        # Modality guidance is disabled by default for DramaBox (modality=1.0);
        # the guider raises if a non-unit modality_scale is requested without it.
        modality = None

        pred = guider(cond, uncond=uncond, ptb=ptb, modality=modality)

        # Re-blend frozen ref tokens BEFORE the Euler step
        pred = post_process_latent(pred, state.denoise_mask, state.clean_latent)

        # Euler step: velocity = (latent - pred) / sigma; new = latent + v * (sigma_next - sigma)
        sigma_val = float(sigma)
        if sigma_val == 0.0:
            # Already at the terminal; nothing to do
            break
        velocity = to_velocity(state.latent, sigma_val, pred)
        dt = float(sigma_next) - sigma_val
        new_latent = state.latent.astype(mx.float32) + velocity.astype(mx.float32) * dt
        state = state.replace(latent=new_latent.astype(state.latent.dtype))

        # Bound the graph at the step boundary
        mx.eval(state.latent)

    return state


__all__ = ["euler_denoising_loop"]
