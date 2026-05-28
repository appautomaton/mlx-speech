"""DramaBox audio patchifier.

Flattens ``(channels, mel_bins)`` into the per-frame patch token dim and
produces the corresponding `[B, 1, T, 2]` start/end timing positions.

Reference: `.references/DramaBox/ltx2/ltx_core/components/patchifiers.py:169-348`
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class AudioLatentShape:
    """``(batch, channels, frames, mel_bins)`` — the unpatchified latent shape."""

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.batch, self.channels, self.frames, self.mel_bins)

    def token_count(self) -> int:
        """Number of tokens after patchify (one per latent time-step)."""
        return self.frames


class AudioPatchifier:
    """Patchify/unpatchify audio latents and emit per-frame timing positions.

    Args:
        sample_rate: source waveform sample rate (16_000).
        hop_length: STFT hop in samples (160).
        audio_latent_downsample_factor: VAE temporal downsample (4).
        is_causal: emit timestamps for the first sample fully available
            within the causal receptive field.
        shift: integer offset applied to latent indices (used by overlapping
            window construction; default 0).
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16_000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
        shift: int = 0,
    ):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self.shift = shift

    # ----- patchify / unpatchify ------------------------------------------

    @staticmethod
    def patchify(latent: mx.array) -> mx.array:
        """``[B, C, T, F] → [B, T, C*F]``.

        Patching order matches ``einops.rearrange("b c t f -> b t (c f)")``:
        the channel index varies slowest within the last dimension.
        """
        if latent.ndim != 4:
            raise ValueError(f"patchify expects 4D [B, C, T, F]; got shape {latent.shape}")
        B, C, T, F = latent.shape
        # transpose to (B, T, C, F), then reshape (B, T, C*F)
        return latent.transpose(0, 2, 1, 3).reshape(B, T, C * F)

    @staticmethod
    def unpatchify(latent: mx.array, channels: int, mel_bins: int) -> mx.array:
        """``[B, T, C*F] → [B, C, T, F]``."""
        if latent.ndim != 3:
            raise ValueError(f"unpatchify expects 3D [B, T, C*F]; got shape {latent.shape}")
        B, T, CF = latent.shape
        if CF != channels * mel_bins:
            raise ValueError(
                f"last dim {CF} != channels*mel_bins = {channels}*{mel_bins} = {channels * mel_bins}"
            )
        # reshape (B, T, C, F), then transpose to (B, C, T, F)
        return latent.reshape(B, T, channels, mel_bins).transpose(0, 2, 1, 3)

    # ----- timing positions ------------------------------------------------

    def _latent_time_in_sec(self, start: int, end: int) -> mx.array:
        """Convert latent indices ``[start, end)`` to timestamps in seconds.

        Mirrors `_get_audio_latent_time_in_sec` from the upstream code.
        """
        idx = mx.arange(start, end, dtype=mx.float32)
        mel_frame = idx * float(self.audio_latent_downsample_factor)
        if self.is_causal:
            causal_offset = 1
            mel_frame = mx.maximum(
                mel_frame + (causal_offset - float(self.audio_latent_downsample_factor)),
                mx.array(0.0, dtype=mx.float32),
            )
        return mel_frame * float(self.hop_length) / float(self.sample_rate)

    def get_patch_grid_bounds(self, shape: AudioLatentShape) -> mx.array:
        """Return per-frame ``(start_s, end_s)`` timings as ``[B, 1, T, 2]``."""
        start = self._latent_time_in_sec(self.shift, shape.frames + self.shift)  # [T]
        end = self._latent_time_in_sec(self.shift + 1, shape.frames + self.shift + 1)  # [T]
        start = mx.broadcast_to(start[None, None, :], (shape.batch, 1, shape.frames))
        end = mx.broadcast_to(end[None, None, :], (shape.batch, 1, shape.frames))
        return mx.stack([start, end], axis=-1)


__all__ = ["AudioLatentShape", "AudioPatchifier"]
