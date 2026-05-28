"""Audio encoder + decoder.

The two towers share PixelNorm/SiLU/CausalConv2d primitives. The encoder
has 2 ResnetBlocks per level (`num_res_blocks=2`); the decoder has 3
(`num_res_blocks + 1`).

Saved key layout:

    encoder.conv_in.conv.{w,b}
    encoder.down.{0..2}.block.{0..1}.conv1.conv.{w,b}
    encoder.down.{0..2}.block.{0..1}.conv2.conv.{w,b}
    encoder.down.{0..2}.block.0.nin_shortcut.conv.{w,b}  (when in!=out)
    encoder.down.{0..1}.downsample.conv.{w,b}            (not at last level)
    encoder.mid.block_1.conv1.conv.{w,b}
    encoder.mid.block_1.conv2.conv.{w,b}
    encoder.mid.block_2.conv1.conv.{w,b}
    encoder.mid.block_2.conv2.conv.{w,b}
    encoder.conv_out.conv.{w,b}                          (out = 2*z_channels)

    decoder.conv_in.conv.{w,b}                           (in = z_channels)
    decoder.mid.block_{1,2}.conv{1,2}.conv.{w,b}
    decoder.up.{0..2}.block.{0..2}.conv{1,2}.conv.{w,b}
    decoder.up.{0..2}.block.0.nin_shortcut.conv.{w,b}    (when in!=out)
    decoder.up.{1..2}.upsample.conv.conv.{w,b}           (not at level 0)
    decoder.conv_out.conv.{w,b}                          (out = out_ch=2)

Reference:
- `.references/DramaBox/ltx2/ltx_core/model/audio_vae/audio_vae.py:59-245`
- `.references/DramaBox/ltx2/ltx_core/model/audio_vae/downsample.py`
- `.references/DramaBox/ltx2/ltx_core/model/audio_vae/upsample.py`
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .causal_conv_2d import CausalConv2d
from .config import AudioVAEConfig
from .pixel_norm import pixel_norm
from .resampling import Downsample, Upsample
from .resnet import ResnetBlock


# --------------------------------------------------------------------------- #
# Mid block (2 resnet blocks; no attention for DramaBox)
# --------------------------------------------------------------------------- #

class _MidBlock(nn.Module):
    """``mid.block_1 → mid.block_2`` (no attention; ``mid_block_add_attention=False``)."""

    def __init__(self, channels: int):
        super().__init__()
        self.block_1 = ResnetBlock(in_channels=channels, out_channels=channels)
        self.block_2 = ResnetBlock(in_channels=channels, out_channels=channels)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.block_1(x)
        return self.block_2(x)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #

class _DownStage(nn.Module):
    """One level of the encoder: list of ResnetBlocks + optional Downsample.

    Saved keys:
        block.{i}.conv{1,2}.conv.{w,b}
        block.0.nin_shortcut.conv.{w,b}     (only when in != out at level entry)
        downsample.conv.{w,b}                (not present at the last level)
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, *, with_downsample: bool):
        super().__init__()
        blocks = []
        ch_in = in_channels
        for _ in range(num_blocks):
            blocks.append(ResnetBlock(in_channels=ch_in, out_channels=out_channels))
            ch_in = out_channels
        # Use a list (MLX serializes children as block.0, block.1, ...)
        self.block = blocks
        if with_downsample:
            self.downsample: Downsample | None = Downsample(out_channels)
        else:
            self.downsample = None

    def __call__(self, x: mx.array) -> mx.array:
        for b in self.block:
            x = b(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class AudioEncoder(nn.Module):
    """Audio VAE encoder: mel-spectrogram → latent (mean only, post-normalize).

    Forward shape: ``[B, T_mel, n_mels, in_channels]`` → ``[B, T_lat, mel_bins_lat, z_channels]``.

    MLX channel-last: we expose the public forward in MLX shape conventions
    and convert at the boundary with the rest of the pipeline (which often
    uses ``[B, C, T, F]``-style shapes).
    """

    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.config = config
        ch = config.ch
        ch_mult = config.ch_mult
        n_res = config.num_res_blocks
        num_resolutions = config.num_resolutions

        self.conv_in = CausalConv2d(config.in_channels, ch, kernel_size=3, stride=1, bias=True)

        # Down stages
        down_stages: list[_DownStage] = []
        block_in = ch
        for level in range(num_resolutions):
            block_out = ch * ch_mult[level]
            down_stages.append(
                _DownStage(
                    in_channels=block_in,
                    out_channels=block_out,
                    num_blocks=n_res,
                    with_downsample=(level != num_resolutions - 1),
                )
            )
            block_in = block_out
        self.down = down_stages

        # Mid block at full depth
        self.mid = _MidBlock(channels=block_in)

        # Output (pre-pixel-norm + SiLU + conv) to 2 * z_channels
        out_channels = 2 * config.z_channels if config.double_z else config.z_channels
        self.conv_out = CausalConv2d(block_in, out_channels, kernel_size=3, stride=1, bias=True)

    def __call__(self, mel: mx.array) -> mx.array:
        """Forward pass returning the encoder's raw output (mean + logvar
        concatenated along channel if ``double_z``)."""
        h = self.conv_in(mel)
        for stage in self.down:
            h = stage(h)
        h = self.mid(h)
        h = pixel_norm(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #

class _UpStage(nn.Module):
    """One level of the decoder: list of ResnetBlocks + optional Upsample.

    Saved keys:
        block.{i}.conv{1,2}.conv.{w,b}                     (i in [0, num_res_blocks])
        block.0.nin_shortcut.conv.{w,b}                    (when in != out at level entry)
        upsample.conv.conv.{w,b}                           (present at non-zero levels)
    """

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int, *, with_upsample: bool):
        super().__init__()
        blocks = []
        ch_in = in_channels
        for _ in range(num_blocks):
            blocks.append(ResnetBlock(in_channels=ch_in, out_channels=out_channels))
            ch_in = out_channels
        self.block = blocks
        if with_upsample:
            self.upsample: Upsample | None = Upsample(out_channels)
        else:
            self.upsample = None

    def __call__(self, x: mx.array) -> mx.array:
        for b in self.block:
            x = b(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class AudioDecoder(nn.Module):
    """Audio VAE decoder: latent → mel-spectrogram.

    Forward shape: ``[B, T_lat, mel_bins_lat, z_channels]`` →
    ``[B, T_mel, n_mels, out_ch]`` (still channel-last).
    """

    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        self.config = config
        ch = config.ch
        ch_mult = config.ch_mult
        n_res = config.num_res_blocks
        num_resolutions = config.num_resolutions

        block_in = ch * ch_mult[-1]  # 512 for DramaBox
        self.conv_in = CausalConv2d(config.z_channels, block_in, kernel_size=3, stride=1, bias=True)
        self.mid = _MidBlock(channels=block_in)

        # Up stages — stored in level-index order (up[0] is the LAST stage)
        # but iterated in reversed level order at forward time.
        up_stages: list[_UpStage] = [None] * num_resolutions  # type: ignore[list-item]
        for level in reversed(range(num_resolutions)):
            block_out = ch * ch_mult[level]
            up_stages[level] = _UpStage(
                in_channels=block_in,
                out_channels=block_out,
                num_blocks=n_res + 1,
                with_upsample=(level != 0),
            )
            block_in = block_out
        self.up = up_stages

        self.conv_out = CausalConv2d(block_in, config.out_ch, kernel_size=3, stride=1, bias=True)

    def __call__(self, latent: mx.array) -> mx.array:
        h = self.conv_in(latent)
        h = self.mid(h)
        for level in reversed(range(self.config.num_resolutions)):
            h = self.up[level](h)
        h = pixel_norm(h)
        h = nn.silu(h)
        h = self.conv_out(h)
        return h


__all__ = ["AudioEncoder", "AudioDecoder"]
