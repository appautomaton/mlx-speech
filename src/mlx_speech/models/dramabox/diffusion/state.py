"""LatentState dataclass and `AudioLatentTools` orchestrator.

The state mirrors the upstream `LatentState` (fields: `latent`,
`denoise_mask`, `positions`, `clean_latent`, `attention_mask`). The tools
build the initial state and handle patchify/unpatchify + the
``clear_conditioning`` step that strips trailing IC-LoRA reference tokens
before VAE decode.

Reference:
- `.references/DramaBox/ltx2/ltx_core/types.py:183-209` — LatentState dataclass
- `.references/DramaBox/ltx2/ltx_core/tools.py:147-190` — AudioLatentTools
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mlx.core as mx

from .patchifier import AudioLatentShape, AudioPatchifier


@dataclass(frozen=True)
class LatentState:
    """State carried through the diffusion sampling loop.

    All tensors share the same leading two dims ``(B, T)``:

    Attributes:
        latent: current noisy/denoised latent ``[B, T, C*F]`` (patchified).
        denoise_mask: per-token denoise strength ``[B, T, 1]`` (1 = full
            denoise, 0 = frozen). For audio the mask is broadcast across
            the patch feature dim.
        positions: per-token positional bounds ``[B, 1, T, 2]`` containing
            ``(start_s, end_s)`` timestamps in seconds.
        clean_latent: the un-noised reference latent (used for re-blending
            frozen tokens after each Euler step).
        attention_mask: optional ``[B, T, T]`` self-attention mask (used by
            IC-LoRA conditioning to gate ref-vs-target attention).
    """

    latent: mx.array
    denoise_mask: mx.array
    positions: mx.array
    clean_latent: mx.array
    attention_mask: mx.array | None = None

    def replace(self, **kwargs) -> "LatentState":
        """Return a copy with the given fields replaced."""
        return replace(self, **kwargs)


class AudioLatentTools:
    """Build initial latent state, patchify in place, clear conditioning.

    Args:
        patchifier: `AudioPatchifier` configured for the target sample-rate
            / hop / downsample.
        target_shape: latent shape ``[B, C, T, F]`` to build the initial
            state for.
    """

    def __init__(self, patchifier: AudioPatchifier, target_shape: AudioLatentShape):
        self.patchifier = patchifier
        self.target_shape = target_shape

    def create_initial_state(self, *, dtype: mx.Dtype = mx.float32) -> LatentState:
        """Create a zero-latent state with per-frame positions and denoise=1."""
        shape = self.target_shape
        zeros = mx.zeros(shape.to_tuple(), dtype=dtype)
        # Mask shape: (B, 1, T, 1) — broadcastable across channels and freq
        mask = mx.ones(
            (shape.batch, 1, shape.frames, 1), dtype=mx.float32,
        )
        positions = self.patchifier.get_patch_grid_bounds(shape)  # [B, 1, T, 2]

        state = LatentState(
            latent=zeros,
            denoise_mask=mask,
            positions=positions,
            clean_latent=zeros,
            attention_mask=None,
        )
        return self.patchify_state(state)

    def patchify_state(self, state: LatentState) -> LatentState:
        """Patchify the 4D fields into the sampler's flat form.

        The patchifier flattens ``(channels, mel_bins) → patch_dim``:
        - ``latent``       (B, C, T, F) → (B, T, C*F)
        - ``clean_latent`` (B, C, T, F) → (B, T, C*F)
        - ``denoise_mask`` (B, 1, T, 1) → (B, T, 1)

        The reference patchifies the mask DIRECTLY at its original
        ``(B, 1, T, 1)`` shape (see `.references/DramaBox/ltx2/ltx_core/tools.py`).
        We do NOT broadcast it to ``(B, C, T, F)`` first — that would change
        the mask shape downstream and break the per-token sigma multiply
        in `timesteps_from_mask(denoise_mask, sigma)`.
        """
        latent = self.patchifier.patchify(state.latent)
        clean = self.patchifier.patchify(state.clean_latent)
        mask = self.patchifier.patchify(state.denoise_mask)
        return state.replace(latent=latent, clean_latent=clean, denoise_mask=mask)

    def unpatchify_state(self, state: LatentState) -> LatentState:
        """Inverse of `patchify_state` for the `latent` and `clean_latent`
        fields. The denoise mask is left in patchified form because we
        rarely need it at unpatchify time."""
        C = self.target_shape.channels
        F = self.target_shape.mel_bins
        latent = self.patchifier.unpatchify(state.latent, C, F)
        clean = self.patchifier.unpatchify(state.clean_latent, C, F)
        return state.replace(latent=latent, clean_latent=clean)

    def clear_conditioning(self, state: LatentState) -> LatentState:
        """Strip trailing IC-LoRA reference tokens from the patchified state.

        After patchify the target sequence is ``num_target`` tokens long. If
        IC-LoRA voice-ref conditioning appended ``num_ref`` extra tokens at
        the end, this method slices the trailing tail back off so downstream
        VAE decoding sees only the target latent.

        Resets ``denoise_mask`` to all-ones and clears ``attention_mask``.
        """
        n = self.target_shape.token_count()
        latent = state.latent[:, :n]
        clean = state.clean_latent[:, :n]
        ones = mx.ones_like(state.denoise_mask)[:, :n]
        positions = state.positions[:, :, :n]
        return LatentState(
            latent=latent,
            denoise_mask=ones,
            positions=positions,
            clean_latent=clean,
            attention_mask=None,
        )


__all__ = ["LatentState", "AudioLatentTools"]
