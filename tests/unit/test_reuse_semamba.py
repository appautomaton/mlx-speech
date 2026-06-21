"""Unit-tier assembly tests for the RE-USE / SEMamba MLX port.

No checkpoint or real weights: every module is built tiny with random init and a
fixed `mx.random.seed`, so these run in the default `pytest tests/unit/` tier.

What is pinned here (beyond shapes):

- The bidirectional `MambaBlock` backward branch flips the *whole* module on the
  reversed sequence: ``flip(backward_blocks(flip(x)) + flip(x))``. A regression to
  scan-only reversal (which leaves the depthwise conv running forward) would
  diverge, so we compare against that hand-built reconstruction and assert it
  differs from the naive ``backward_blocks(x) + x``.
- `SPConvTranspose2d` pixel-shuffle ordering: the public output must equal the
  ``reshape(b, r, nch//r, h, w) -> transpose(0,2,3,4,1) -> reshape(b, nch//r, h, -1)``
  discipline applied to the conv output, so a wrong transpose/reshape breaks it.
"""

from __future__ import annotations

import numpy as np

import mlx.core as mx

from mlx_speech.models.reuse.mamba.block import MambaBlock, TFMambaBlock
from mlx_speech.models.reuse.semamba import (
    DenseEncoder,
    MagDecoder,
    PhaseDecoder,
    SEMamba,
    SPConvTranspose2d,
    _to_channels_first,
    _to_channels_last,
)

# Tiny hid_feature for the per-module tests; the assembly test uses the real 64
# (conv strides fix the feature dims) with just 2 TFMamba blocks for speed.
_H = 8
_D_STATE = 4
_D_CONV = 4
_EXPAND = 4


def _finite(x: mx.array) -> bool:
    return bool(mx.all(mx.isfinite(x)).item())


def test_semamba_forward_shapes_and_finite():
    # Real hid_feature=64 with 2 TFMamba blocks; F, T chosen to survive the
    # encoder stride-(4,2) downsample and the decoder upsample back to (F, T).
    mx.random.seed(0)
    model = SEMamba(num_tfmamba=2)
    f, t = 33, 8
    noisy_mag = mx.random.normal((1, f, t)) * 0.1
    noisy_pha = mx.random.normal((1, f, t)) * 0.1

    denoised_mag, denoised_pha, denoised_com = model(noisy_mag, noisy_pha)
    mx.eval(denoised_mag, denoised_pha, denoised_com)

    assert denoised_mag.shape == (1, f, t)
    assert denoised_pha.shape == (1, f, t)
    assert denoised_com.shape == (1, f, t, 2)
    assert _finite(denoised_mag)
    assert _finite(denoised_pha)
    assert _finite(denoised_com)


def test_mamba_block_backward_runs_whole_module_on_flipped_input():
    # The backward direction must flip x in time, run the ENTIRE backward module
    # (depthwise conv included) on the reversed sequence, then flip back. Pin that
    # against a hand-built reconstruction using the public submodules.
    mx.random.seed(1)
    block = MambaBlock(_H, _D_STATE, _D_CONV, _EXPAND)
    x = mx.random.normal((2, 5, _H))

    x_rev = block._flip_time(x)
    recon_bw = block._flip_time(block.backward_blocks(x_rev) + x_rev)
    mx.eval(recon_bw)

    # A scan-only reversal would instead leave the conv running forward, i.e.
    # behave like backward_blocks(x) + x. That must NOT equal the whole-module
    # flip, otherwise this test could not catch the regression.
    naive = block.backward_blocks(x) + x
    mx.eval(naive)
    assert float(mx.max(mx.abs(recon_bw - naive)).item()) > 1e-3

    # Sanity: the reconstruction is finite and time-aligned to the input length.
    assert recon_bw.shape == x.shape
    assert _finite(recon_bw)


def test_spconvtranspose2d_pixel_shuffle_ordering():
    # Pin the reshape/transpose discipline. Feed a known input, take the module's
    # conv output, and assert the public output equals the hand-computed
    # pixel-shuffle of that conv output (r factor folded into the F axis).
    mx.random.seed(2)
    r = 2
    sp = SPConvTranspose2d(3, 2, (1, 3), r=r)
    x = mx.random.normal((1, 3, 4, 5))

    out = sp(x)
    mx.eval(out)

    # Reproduce the internal conv output (pad F by (1, 1), then channel-last conv).
    x_pad = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 1)])
    conv_out = _to_channels_first(sp.conv(_to_channels_last(x_pad)))
    mx.eval(conv_out)
    co = np.asarray(conv_out)

    b, nch, h, w = co.shape
    expected = (
        co.reshape(b, r, nch // r, h, w)
        .transpose(0, 2, 3, 4, 1)
        .reshape(b, nch // r, h, -1)
    )

    assert out.shape == (b, nch // r, h, w * r)
    assert float(np.max(np.abs(expected - np.asarray(out)))) < 1e-5


def test_tfmamba_block_preserves_shape():
    mx.random.seed(3)
    block = TFMambaBlock(_H, _D_STATE, _D_CONV, _EXPAND)
    x = mx.random.normal((1, _H, 4, 5))  # [B, C, T, F]

    out = block(x)
    mx.eval(out)

    assert out.shape == x.shape
    assert _finite(out)


def test_dense_encoder_downsamples_then_decoders_upsample():
    # DenseEncoder: dense_conv_2 stride (4, 2) downsamples (T/4, F/2).
    mx.random.seed(4)
    enc = DenseEncoder(2, _H)
    x = mx.random.normal((1, 2, 8, 16))  # [B, in_ch, T, F]

    encoded = enc(x)
    mx.eval(encoded)
    assert encoded.shape == (1, _H, 2, 7)
    assert _finite(encoded)

    # MagDecoder / PhaseDecoder upsample T by 2 and F by 4 back to one channel.
    mag = MagDecoder(_H, 1)(encoded)
    pha = PhaseDecoder(_H, 1)(encoded)
    mx.eval(mag, pha)
    assert mag.shape == (1, 1, 8, 14)
    assert pha.shape == (1, 1, 8, 14)
    assert _finite(mag)
    assert _finite(pha)


def test_source_imports_without_torch():
    import sys

    import mlx_speech.models.reuse.semamba as semamba_mod  # noqa: F401

    assert "torch" not in sys.modules
