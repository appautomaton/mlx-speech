"""Target latent shape for the DramaBox audio DiT.

Implements the warm-server frame rounding (`inference_server.py:325-335`) and
the latent-shape derivation (`types.py:128-164`).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioTargetShape:
    """Latent target shape ``[batch, channels=8, frames, mel_bins=16]``.

    `frames` is the number of latent time steps after the warm-server
    rounding rule. Patchification later flattens `(channels, mel_bins)` →
    ``128``, giving final shape ``[batch, frames, 128]``.
    """

    batch: int
    channels: int
    frames: int
    mel_bins: int

    def to_tuple(self) -> tuple[int, int, int, int]:
        return (self.batch, self.channels, self.frames, self.mel_bins)


def target_shape_from_duration(
    duration_s: float,
    *,
    batch: int = 1,
    fps: float = 25.0,
    channels: int = 8,
    mel_bins: int = 16,
    sample_rate: int = 16_000,
    hop_length: int = 160,
    audio_latent_downsample_factor: int = 4,
) -> AudioTargetShape:
    """Compute the audio latent target shape for a given duration.

    Algorithm:
        n_frames = round(duration_s * fps) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1   # align to 8k+1
        latents_per_second = sample_rate / hop_length / downsample
        audio_frames = round(n_frames / fps * latents_per_second)

    For DramaBox the configuration yields
    ``latents_per_second = 16000 / 160 / 4 = 25.0``, so
    ``audio_frames == n_frames``.
    """
    n_frames = int(round(duration_s * fps)) + 1
    n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1

    latents_per_second = float(sample_rate) / float(hop_length) / float(audio_latent_downsample_factor)
    audio_frames = int(round(n_frames / fps * latents_per_second))

    return AudioTargetShape(
        batch=batch,
        channels=channels,
        frames=audio_frames,
        mel_bins=mel_bins,
    )


__all__ = ["AudioTargetShape", "target_shape_from_duration"]
