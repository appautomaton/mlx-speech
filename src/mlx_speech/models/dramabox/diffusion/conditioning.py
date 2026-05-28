"""Reference-latent conditioning for DramaBox DiT sampling."""

from __future__ import annotations

import mlx.core as mx

from .patchifier import AudioLatentShape, AudioPatchifier
from .state import LatentState


def _validate_patchified_state(state: LatentState) -> None:
    if state.latent.ndim != 3:
        raise ValueError(f"state.latent must be patchified [B, T, D]; got {state.latent.shape}")
    if state.clean_latent.shape != state.latent.shape:
        raise ValueError(
            f"state.clean_latent shape {state.clean_latent.shape} must match latent {state.latent.shape}"
        )
    if state.denoise_mask.shape[:2] != state.latent.shape[:2] or state.denoise_mask.shape[-1] != 1:
        raise ValueError(
            f"state.denoise_mask must be [B, T, 1] matching latent; got {state.denoise_mask.shape}"
        )
    if state.positions.shape[:3] != (state.latent.shape[0], 1, state.latent.shape[1]):
        raise ValueError(
            f"state.positions must be [B, 1, T, 2] matching latent; got {state.positions.shape}"
        )


def _build_asymmetric_attention_mask(
    *,
    batch: int,
    target_tokens: int,
    ref_tokens: int,
    dtype: mx.Dtype,
) -> mx.array:
    total = target_tokens + ref_tokens
    target_rows = mx.zeros((batch, 1, target_tokens, total), dtype=dtype)
    ref_to_target = mx.full(
        (batch, 1, ref_tokens, target_tokens),
        mx.array(mx.finfo(dtype).min, dtype=dtype),
        dtype=dtype,
    )
    ref_to_ref = mx.zeros((batch, 1, ref_tokens, ref_tokens), dtype=dtype)
    ref_rows = mx.concatenate([ref_to_target, ref_to_ref], axis=-1)
    return mx.concatenate([target_rows, ref_rows], axis=-2)


def apply_reference_latent(
    state: LatentState,
    ref_latent: mx.array,
    *,
    patchifier: AudioPatchifier | None = None,
    position_offset_s: float = 0.5,
) -> LatentState:
    """Append frozen reference latent tokens and build the asymmetric self-attn mask.

    Args:
        state: patchified target state from `AudioLatentTools.create_initial_state`.
        ref_latent: unpatchified reference latent `[B, C, T_ref, F]`.
        patchifier: patchifier matching the target/reference latent layout.
        position_offset_s: seconds added to reference token positions.
    """
    _validate_patchified_state(state)
    if ref_latent.ndim != 4:
        raise ValueError(f"ref_latent must be [B, C, T, F]; got {ref_latent.shape}")
    if ref_latent.shape[0] != state.latent.shape[0]:
        raise ValueError(
            f"ref batch {ref_latent.shape[0]} must match target batch {state.latent.shape[0]}"
        )

    patchifier = patchifier or AudioPatchifier()
    ref_tokens = patchifier.patchify(ref_latent).astype(state.latent.dtype)
    if ref_tokens.shape[-1] != state.latent.shape[-1]:
        raise ValueError(
            f"ref token dim {ref_tokens.shape[-1]} must match target token dim {state.latent.shape[-1]}"
        )

    batch = state.latent.shape[0]
    target_count = state.latent.shape[1]
    ref_count = ref_tokens.shape[1]

    ref_mask = mx.zeros((batch, ref_count, 1), dtype=state.denoise_mask.dtype)
    ref_shape = AudioLatentShape(
        batch=batch,
        channels=ref_latent.shape[1],
        frames=ref_latent.shape[2],
        mel_bins=ref_latent.shape[3],
    )
    ref_positions = patchifier.get_patch_grid_bounds(ref_shape) + float(position_offset_s)
    attention_mask = _build_asymmetric_attention_mask(
        batch=batch,
        target_tokens=target_count,
        ref_tokens=ref_count,
        dtype=state.latent.dtype,
    )

    return LatentState(
        latent=mx.concatenate([state.latent, ref_tokens], axis=1),
        denoise_mask=mx.concatenate([state.denoise_mask, ref_mask], axis=1),
        positions=mx.concatenate([state.positions, ref_positions], axis=2),
        clean_latent=mx.concatenate([state.clean_latent, ref_tokens], axis=1),
        attention_mask=attention_mask,
    )


__all__ = ["apply_reference_latent"]
