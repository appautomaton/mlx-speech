"""VAE config (mirrors `audio_vae.model.params.ddconfig` from the checkpoint)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioVAEConfig:
    """Static VAE config recovered from `audio-components` metadata.

    For DramaBox the values are fixed; we expose them as defaults so the
    construction code reads cleanly.
    """

    in_channels: int = 2          # stereo
    out_ch: int = 2
    z_channels: int = 8           # latent channels
    ch: int = 128                 # base channels
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    double_z: bool = True         # encoder predicts 2*z_channels (mean + logvar)
    causality_axis: str = "height"  # height = time for audio mel
    norm_type: str = "pixel"
    sampling_rate: int = 16_000
    mel_bins: int = 64
    mel_hop_length: int = 160
    n_fft: int = 1024
    audio_latent_downsample_factor: int = 4  # extra temporal downsample inside the patchifier (post-VAE)

    @property
    def num_resolutions(self) -> int:
        return len(self.ch_mult)

    @property
    def base_block_channels(self) -> int:
        """Channels at the bottom of the down/up tower (after ch_mult[-1])."""
        return self.ch * self.ch_mult[-1]


__all__ = ["AudioVAEConfig"]
