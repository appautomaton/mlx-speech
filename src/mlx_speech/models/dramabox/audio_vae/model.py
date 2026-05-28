"""`AudioVAE` container.

Holds the encoder, decoder, and per-channel statistics; exposes high-level
``encode`` / ``decode`` methods that handle the post-encoder normalize and
pre-decoder un-normalize.

The patchifier translates between latent shape ``[B, C, T, F]`` and the
patchified ``[B, T, C*F]`` form used by the diffusion sampler.

Reference: `.references/DramaBox/ltx2/ltx_core/model/audio_vae/audio_vae.py:189-245, 384-484`
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..diffusion.patchifier import AudioPatchifier
from .config import AudioVAEConfig
from .encoder_decoder import AudioDecoder, AudioEncoder
from .per_channel_statistics import PerChannelStatistics


class AudioVAE(nn.Module):
    """Encoder + decoder + per-channel statistics, plus a patchifier.

    Channel-last MLX convention internally. Public methods accept and return
    PyTorch-style ``[B, C, T, F]`` shapes for compatibility with the
    upstream code we mirror.
    """

    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoder(config)
        self.decoder = AudioDecoder(config)
        # Latent mel-bins after encoder downsampling: input_mel_bins //
        # 2**(num_resolutions - 1). For DramaBox: 64 / 4 = 16. The patchified
        # last dim is z_channels * latent_mel_bins (8 * 16 = 128).
        latent_mel_bins = config.mel_bins // (2 ** (config.num_resolutions - 1))
        pcs_dim = config.z_channels * latent_mel_bins
        self.per_channel_statistics = PerChannelStatistics(dim=pcs_dim)
        self.patchifier = AudioPatchifier(
            sample_rate=config.sampling_rate,
            hop_length=config.mel_hop_length,
            audio_latent_downsample_factor=config.audio_latent_downsample_factor,
            is_causal=True,
        )

    # ----- public helpers --------------------------------------------------

    def encode(self, mel: mx.array) -> mx.array:
        """Encode mel ``[B, C, T_mel, n_mels]`` → normalized latent ``[B, C_lat=8, T_lat, mel_bins_lat=16]``.

        Steps:
            1. Encoder forward → ``[B, 2*z_channels, T_lat, mel_bins_lat]``
            2. Chunk first half (means) → ``[B, z_channels, ...]``
            3. Patchify → ``[B, T_lat, z_channels * mel_bins_lat = 128]``
            4. Per-channel-statistics normalize
            5. Unpatchify back to ``[B, z_channels, T_lat, mel_bins_lat]``
        """
        # PyTorch shape (B, C, T, F) → MLX channel-last (B, T, F, C)
        mel_cl = _bcthf_to_btfc(mel)
        raw = self.encoder(mel_cl)  # (B, T_lat, mel_bins_lat, 2*z_channels)
        # Take the first half (means) along the channel (last) axis
        z_ch = self.config.z_channels
        means_cl = raw[..., :z_ch]  # (B, T_lat, mel_bins_lat, z_channels)
        means = _btfc_to_bcthf(means_cl)
        # Patchify and normalize
        patched = self.patchifier.patchify(means)  # (B, T_lat, C*F)
        normed = self.per_channel_statistics.normalize(patched)
        # Unpatchify back
        return self.patchifier.unpatchify(
            normed,
            channels=self.config.z_channels,
            mel_bins=means.shape[3],
        )

    def decode(self, latent: mx.array) -> mx.array:
        """Decode normalized latent ``[B, z_channels, T_lat, mel_bins_lat]`` →
        mel ``[B, out_ch=2, T_mel, n_mels=64]``.

        Steps:
            1. Patchify → un-normalize → unpatchify
            2. Decoder forward
        """
        # Un-normalize via patchify → ops → unpatchify
        patched = self.patchifier.patchify(latent)
        denormed = self.per_channel_statistics.un_normalize(patched)
        unp = self.patchifier.unpatchify(
            denormed,
            channels=latent.shape[1],
            mel_bins=latent.shape[3],
        )
        # Decoder runs channel-last
        cl = _bcthf_to_btfc(unp)
        out_cl = self.decoder(cl)
        return _btfc_to_bcthf(out_cl)


# --------------------------------------------------------------------------- #
# Channel-format helpers (PyTorch [B,C,T,F] ↔ MLX [B,T,F,C])
# --------------------------------------------------------------------------- #

def _bcthf_to_btfc(x: mx.array) -> mx.array:
    """``[B, C, T, F]`` → ``[B, T, F, C]``."""
    return x.transpose(0, 2, 3, 1)


def _btfc_to_bcthf(x: mx.array) -> mx.array:
    """``[B, T, F, C]`` → ``[B, C, T, F]``."""
    return x.transpose(0, 3, 1, 2)


__all__ = ["AudioVAE"]
