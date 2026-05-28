"""Unit tests for the DramaBox AudioVAE primitives.

Tests run against small synthetic shapes for the conv/norm/resnet pieces;
the full VAE shape test uses a tiny config (ch=4, single down/up level)
so it builds quickly without any checkpoints.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.audio_vae import AudioVAE, AudioVAEConfig
from mlx_speech.models.dramabox.audio_vae.causal_conv_2d import CausalConv2d
from mlx_speech.models.dramabox.audio_vae.encoder_decoder import (
    AudioDecoder,
    AudioEncoder,
)
from mlx_speech.models.dramabox.audio_vae.per_channel_statistics import (
    PerChannelStatistics,
)
from mlx_speech.models.dramabox.audio_vae.pixel_norm import pixel_norm
from mlx_speech.models.dramabox.audio_vae.resampling import Downsample, Upsample
from mlx_speech.models.dramabox.audio_vae.resnet import ResnetBlock


# --------------------------------------------------------------------------- #
# CausalConv2d — height-axis (time) causality
# --------------------------------------------------------------------------- #

def test_causal_conv2d_preserves_height_width():
    """Kernel 3, stride 1, dilation 1 → output H == input H, W == input W."""
    conv = CausalConv2d(2, 4, kernel_size=3, stride=1, bias=True)
    x = mx.random.normal((1, 8, 16, 2), dtype=mx.float32)  # B, H, W, C
    y = conv(x)
    assert y.shape == (1, 8, 16, 4)


def test_causal_conv2d_kernel1_is_pointwise():
    """1×1 conv == matmul along channels; output spatial size unchanged."""
    conv = CausalConv2d(3, 6, kernel_size=1, stride=1, bias=True)
    x = mx.random.normal((1, 4, 5, 3), dtype=mx.float32)
    y = conv(x)
    assert y.shape == (1, 4, 5, 6)


def test_causal_conv2d_causality_along_height():
    """Output at height index h depends only on input rows 0..h.

    Construct an input that is zero everywhere except row `h`. The output
    at row < h must be zero (no leakage from future rows)."""
    conv = CausalConv2d(1, 1, kernel_size=3, stride=1, bias=False)
    # Set the conv weights to all 1s (so output is sum of windowed input)
    # MLX `nn.Conv2d` weight shape: (out_C, k_h, k_w, in_C). Set to 1.
    conv.conv.weight = mx.ones_like(conv.conv.weight)
    H = 5
    # Input: zero everywhere except row 3 (the 4th row) which is all-ones.
    x = mx.zeros((1, H, 3, 1), dtype=mx.float32)
    x = mx.concatenate(
        [
            mx.zeros((1, 3, 3, 1), dtype=mx.float32),
            mx.ones((1, 1, 3, 1), dtype=mx.float32),
            mx.zeros((1, 1, 3, 1), dtype=mx.float32),
        ],
        axis=1,
    )
    y = conv(x)
    # Rows 0..2 should be zero (no future-input dependence)
    assert mx.allclose(y[:, :3], mx.zeros_like(y[:, :3]), atol=1e-5).item()
    # Row 3 receives row-3 input → non-zero
    assert float(mx.abs(y[:, 3]).sum()) > 0.0


# --------------------------------------------------------------------------- #
# PixelNorm
# --------------------------------------------------------------------------- #

def test_pixel_norm_normalizes_along_channel():
    """RMS along the last (channel) axis should be 1 after pixel_norm."""
    x = mx.array([[[[1.0, 2.0, 2.0, 4.0]]]], dtype=mx.float32)  # (1,1,1,4)
    y = pixel_norm(x, eps=0.0)
    rms = float(mx.sqrt(mx.mean(y * y, axis=-1)))
    assert rms == pytest.approx(1.0, abs=1e-5)


def test_pixel_norm_preserves_dtype():
    x = mx.random.normal((1, 4, 4, 8), dtype=mx.bfloat16)
    y = pixel_norm(x)
    assert y.dtype == mx.bfloat16


# --------------------------------------------------------------------------- #
# ResnetBlock
# --------------------------------------------------------------------------- #

def test_resnet_block_same_channels():
    block = ResnetBlock(in_channels=8, out_channels=8)
    x = mx.random.normal((1, 4, 6, 8), dtype=mx.float32)
    y = block(x)
    assert y.shape == x.shape


def test_resnet_block_channel_expansion():
    block = ResnetBlock(in_channels=4, out_channels=8)
    x = mx.random.normal((1, 4, 6, 4), dtype=mx.float32)
    y = block(x)
    assert y.shape == (1, 4, 6, 8)


# --------------------------------------------------------------------------- #
# Downsample / Upsample
# --------------------------------------------------------------------------- #

def test_downsample_halves_spatial():
    down = Downsample(in_channels=4)
    x = mx.random.normal((1, 8, 8, 4), dtype=mx.float32)
    y = down(x)
    # Stride 2 with the custom pad: H 8→4, W 8→4 (approximately).
    # Verify it's roughly half size.
    assert y.shape[1] == 4
    assert y.shape[2] == 4
    assert y.shape[3] == 4  # channels unchanged


def test_upsample_doubles_spatial():
    """Upsample increases spatial size; after the H-axis drop, H = 2H-1."""
    up = Upsample(in_channels=4)
    x = mx.random.normal((1, 4, 4, 4), dtype=mx.float32)
    y = up(x)
    # 2H = 8, drop first row → 7
    assert y.shape[1] == 7
    # 2W = 8
    assert y.shape[2] == 8


# --------------------------------------------------------------------------- #
# PerChannelStatistics
# --------------------------------------------------------------------------- #

def test_per_channel_stats_roundtrip():
    pcs = PerChannelStatistics(dim=4)
    pcs.mean_of_means = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float32)
    pcs.std_of_means = mx.array([0.5, 1.0, 2.0, 4.0], dtype=mx.float32)
    x = mx.random.normal((1, 8, 4), dtype=mx.float32)
    normed = pcs.normalize(x)
    back = pcs.un_normalize(normed)
    assert mx.allclose(back, x, atol=1e-5).item()


# --------------------------------------------------------------------------- #
# Full VAE (tiny config, no checkpoint)
# --------------------------------------------------------------------------- #

def test_audio_vae_decode_shape():
    """Tiny VAE built from default-ish config: decode reasonable shape."""
    # Use a smaller config than DramaBox to keep the test fast: 1 down/up
    # level and small channels.
    cfg = AudioVAEConfig(
        in_channels=2, out_ch=2, z_channels=4, ch=8, ch_mult=(1, 2),
        num_res_blocks=1, double_z=True, mel_bins=8,
    )
    vae = AudioVAE(cfg)
    # Latent shape: [B=1, z_ch=4, T_lat=8, mel_bins_lat=8/2=4]
    latent_mel_bins = cfg.mel_bins // (2 ** (cfg.num_resolutions - 1))
    latent = mx.random.normal((1, 4, 8, latent_mel_bins), dtype=mx.float32)
    out = vae.decode(latent)
    # Should be 4D with out_ch=2 channels
    assert out.shape[0] == 1
    assert out.shape[1] == 2  # out_ch
    # Spatial dims are 2× the input (single upsample stage in this tiny config)
    # Don't assert exact T_mel size since the upsample's height-drop affects it.
    assert out.ndim == 4


def test_audio_vae_encode_shape():
    cfg = AudioVAEConfig(
        in_channels=2, out_ch=2, z_channels=4, ch=8, ch_mult=(1, 2),
        num_res_blocks=1, double_z=True, mel_bins=8,
    )
    vae = AudioVAE(cfg)
    mel = mx.random.normal((1, 2, 8, 8), dtype=mx.float32)  # B, C, T, F=8 matches cfg.mel_bins
    latent = vae.encode(mel)
    # Latent has z_channels=4
    assert latent.shape[0] == 1
    assert latent.shape[1] == 4  # z_channels
    # Spatial dims halved by 1 down-stage
    assert latent.ndim == 4


def test_audio_vae_default_config_matches_dramabox():
    """The default AudioVAEConfig() should match DramaBox values."""
    cfg = AudioVAEConfig()
    assert cfg.in_channels == 2
    assert cfg.z_channels == 8
    assert cfg.ch == 128
    assert cfg.ch_mult == (1, 2, 4)
    assert cfg.num_res_blocks == 2
    assert cfg.sampling_rate == 16_000
    assert cfg.mel_bins == 64
    assert cfg.mel_hop_length == 160
    assert cfg.n_fft == 1024
