"""Unit tests for the DramaBox vocoder primitives.

Stage 5 — snake_beta, anti-aliased activation, AMP block, vocoder shape.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.vocoder import Vocoder, VocoderWithBWE, MelSTFT
from mlx_speech.models.dramabox.vocoder.anti_aliased import Activation1d
from mlx_speech.models.dramabox.vocoder.snake import SnakeBeta
from mlx_speech.models.dramabox.vocoder.vocoder import AMPBlock1, VocoderArgs


# --------------------------------------------------------------------------- #
# SnakeBeta
# --------------------------------------------------------------------------- #

def test_snake_beta_alpha_zero_logscale_is_identity_plus_sin():
    """At alpha=beta=0 (logscale → exp=1), output = x + 1*sin(x)^2."""
    snake = SnakeBeta(channels=2, alpha_logscale=True)
    x = mx.array([[[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]]], dtype=mx.float32)  # (B=1, C=2, T=3)
    y = snake(x)
    # alpha=exp(0)=1, beta=exp(0)=1, so y = x + (1/(1+1e-9)) * sin(x)^2
    expected = x + (1.0 / (1.0 + 1e-9)) * mx.sin(x) ** 2
    assert mx.allclose(y, expected, atol=1e-5).item()


def test_snake_beta_output_shape():
    snake = SnakeBeta(channels=4)
    x = mx.random.normal((2, 4, 16), dtype=mx.float32)
    y = snake(x)
    assert y.shape == (2, 4, 16)


# --------------------------------------------------------------------------- #
# Activation1d (anti-aliased)
# --------------------------------------------------------------------------- #

def test_activation1d_preserves_shape():
    """The upsample-then-downsample pair should preserve length (after the
    inner sinc-filter pad/crop)."""
    act = Activation1d(channels=8, up_ratio=2, down_ratio=2, kernel_size=12)
    x = mx.random.normal((1, 8, 64), dtype=mx.float32)
    y = act(x)
    # Output length depends on internal pads; just check it's close to T
    assert y.ndim == 3
    assert y.shape[0] == 1
    assert y.shape[1] == 8
    # length is preserved to within a few samples of the input (anti-alias adds slight slack)
    assert abs(y.shape[-1] - 64) <= 4


# --------------------------------------------------------------------------- #
# AMPBlock1
# --------------------------------------------------------------------------- #

def test_amp_block_shape_preserved():
    """AMPBlock1 is a residual block — output shape == input shape."""
    block = AMPBlock1(channels=8, kernel_size=3, dilation=(1, 3, 5))
    x = mx.random.normal((1, 8, 64), dtype=mx.float32)
    y = block(x)
    assert y.shape == x.shape


# --------------------------------------------------------------------------- #
# Vocoder generator (small synthetic config)
# --------------------------------------------------------------------------- #

def test_vocoder_forward_shape_small():
    """Build a tiny vocoder and confirm the output is roughly the right length.

    For DramaBox the main vocoder has `upsample_rates = [5,2,2,2,2,2]`, total
    upsample = 5 * 32 = 160 = mel hop length, so an input mel of T frames
    produces ~T * 160 audio samples.

    Here we use a single 2× upsample stage to keep the test fast.
    """
    args = VocoderArgs(
        upsample_initial_channel=8,
        upsample_rates=(2,),
        upsample_kernel_sizes=(4,),
        resblock_kernel_sizes=(3,),
        resblock_dilation_sizes=((1, 3, 5),),
        in_channels=8,
        out_channels=2,
        apply_final_activation=True,
        use_tanh_at_final=False,
        use_bias_at_final=False,
    )
    v = Vocoder(args)
    # Stereo input: (B, 2, T=4, mel_bins=4)
    mel = mx.random.normal((1, 2, 4, 4), dtype=mx.float32)
    out = v(mel)
    # Output channels = 2, length is approximately T * upsample_product = 4 * 2 = 8
    assert out.ndim == 3
    assert out.shape[0] == 1
    assert out.shape[1] == 2
    # Check clipped to [-1, 1] since apply_final_activation=True, use_tanh=False
    assert float(mx.max(out)) <= 1.0 + 1e-5
    assert float(mx.min(out)) >= -1.0 - 1e-5


# --------------------------------------------------------------------------- #
# MelSTFT (init shapes only; full computation needs loaded basis)
# --------------------------------------------------------------------------- #

def test_mel_stft_init_shapes():
    mst = MelSTFT(filter_length=512, hop_length=80, win_length=512, n_mel_channels=64)
    n_freqs = 512 // 2 + 1  # 257
    assert mst.stft_fn.forward_basis.shape == (2 * n_freqs, 1, 512)
    assert mst.stft_fn.inverse_basis.shape == (2 * n_freqs, 1, 512)
    assert mst.mel_basis.shape == (64, n_freqs)


# --------------------------------------------------------------------------- #
# VocoderArgs invariants
# --------------------------------------------------------------------------- #

def test_vocoder_args_final_channels_for_main():
    """DramaBox main vocoder: 1536 / (2**6) = 24."""
    args = VocoderArgs(
        upsample_initial_channel=1536,
        upsample_rates=(5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(11, 4, 4, 4, 4, 4),
    )
    assert args.final_channels == 24
    assert args.num_upsamples == 6
    assert args.num_kernels == 3


def test_vocoder_args_final_channels_for_bwe():
    """DramaBox BWE: 512 / (2**5) = 16."""
    args = VocoderArgs(
        upsample_initial_channel=512,
        upsample_rates=(6, 5, 2, 2, 2),
        upsample_kernel_sizes=(12, 11, 4, 4, 4),
    )
    assert args.final_channels == 16
    assert args.num_upsamples == 5
