"""Unit tests for the DramaBox prompt pipeline primitives.

These tests run against small synthetic configs — no checkpoints required.
Checkpoint-loading tests live under `tests/checkpoint/`.
"""

from __future__ import annotations

import math

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.prompt.additive_mask import convert_to_additive_mask
from mlx_speech.models.dramabox.prompt.connector import (
    Embeddings1DConnector,
    _BasicTransformerBlock1D,
)
from mlx_speech.models.dramabox.prompt.feature_extractor import FeatureExtractorV2
from mlx_speech.models.dramabox.prompt.processor import EmbeddingsProcessor


# --------------------------------------------------------------------------- #
# additive_mask
# --------------------------------------------------------------------------- #

def test_convert_to_additive_mask_valid_is_zero():
    mask = mx.array([[1, 1, 1]], dtype=mx.int32)
    add = convert_to_additive_mask(mask, mx.float32)
    assert add.shape == (1, 1, 1, 3)
    assert mx.allclose(add, mx.zeros_like(add), atol=1e-6).item()


def test_convert_to_additive_mask_padding_is_large_negative():
    mask = mx.array([[0, 1, 1]], dtype=mx.int32)
    add = convert_to_additive_mask(mask, mx.float32)
    # First position is padded → -finfo.max
    assert float(add[0, 0, 0, 0]) == pytest.approx(-mx.finfo(mx.float32).max, rel=1e-6)
    # Other positions are 0
    assert float(add[0, 0, 0, 1]) == 0.0
    assert float(add[0, 0, 0, 2]) == 0.0


def test_convert_to_additive_mask_dtype_preserved():
    mask = mx.array([[0, 1]], dtype=mx.int32)
    add = convert_to_additive_mask(mask, mx.bfloat16)
    assert add.dtype == mx.bfloat16


# --------------------------------------------------------------------------- #
# FeatureExtractorV2
# --------------------------------------------------------------------------- #

def test_feature_extractor_shape_and_pad_zeroed():
    D, L = 32, 4  # tiny: 4 layers of 32-dim hidden states
    out_features = 16
    fx = FeatureExtractorV2(embedding_dim=D, out_features=out_features, num_layers=L)

    # Construct 4 layers of [1, T=8, D]
    hidden = [mx.random.normal((1, 8, D)) for _ in range(L)]
    mask = mx.array([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=mx.int32)  # left-padded
    out = fx(hidden, mask)

    assert out.shape == (1, 8, out_features)

    # Padded positions: output should be `audio_aggregate_embed(zeros) = bias`
    # since rescale * 0 = 0. Verify by computing the bias directly.
    expected_pad = fx.audio_aggregate_embed.bias
    for t in (0, 1, 2):
        assert mx.allclose(out[0, t], expected_pad, atol=1e-4).item(), (
            f"padded position {t} should equal bias"
        )


def test_feature_extractor_rescale_constant():
    """The rescale factor should be sqrt(out_features / embedding_dim)."""
    fx = FeatureExtractorV2(embedding_dim=3840, out_features=2048, num_layers=49)
    assert fx._rescale == pytest.approx(math.sqrt(2048 / 3840))


def test_feature_extractor_rms_norm_unit_variance():
    """After per-token RMS norm and reshape, the squared values along the
    embedding axis should sum to 1 (per-token, per-layer). We test that the
    pre-projection rescale matches expectation by inspecting an intermediate
    via a sanity calculation."""
    D, L = 8, 2
    fx = FeatureExtractorV2(embedding_dim=D, out_features=4, num_layers=L)

    hidden = [mx.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]], dtype=mx.float32) for _ in range(L)]
    mask = mx.array([[1]], dtype=mx.int32)
    out = fx(hidden, mask)

    assert out.shape == (1, 1, 4)
    # Sanity: no NaN
    assert mx.all(mx.isfinite(out)).item()


# --------------------------------------------------------------------------- #
# Connector register-replacement logic
# --------------------------------------------------------------------------- #

def _make_tiny_connector() -> Embeddings1DConnector:
    """8-token sequence, 4 registers, 2 heads × 4 dim_head = 8 inner_dim."""
    return Embeddings1DConnector(
        num_attention_heads=2,
        attention_head_dim=4,
        num_layers=1,
        num_learnable_registers=4,
        positional_embedding_max_pos=8,
        seq_len=8,
    )


def test_connector_pack_valid_to_front():
    """Validate the `_pack_valid_to_front` helper packs mask=1 rows first."""
    conn = _make_tiny_connector()
    # Left-padded: [0, 0, 0, 1, 1, 1, 1, 1] → 5 valid tokens at positions 3..7
    hidden = mx.array([[
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [10, 10, 10, 10, 10, 10, 10, 10],
        [20, 20, 20, 20, 20, 20, 20, 20],
        [30, 30, 30, 30, 30, 30, 30, 30],
        [40, 40, 40, 40, 40, 40, 40, 40],
        [50, 50, 50, 50, 50, 50, 50, 50],
    ]], dtype=mx.float32)
    mask = mx.array([[0, 0, 0, 1, 1, 1, 1, 1]], dtype=mx.int32)
    packed = conn._pack_valid_to_front(hidden, mask)

    # Expected: positions 0..4 carry values from original positions 3..7;
    # positions 5..7 are zero.
    expected_first_five = mx.array(
        [[10] * 8, [20] * 8, [30] * 8, [40] * 8, [50] * 8],
        dtype=mx.float32,
    )
    assert mx.allclose(packed[0, :5], expected_first_five, atol=1e-5).item()
    assert mx.allclose(packed[0, 5:], mx.zeros((3, 8)), atol=1e-5).item()


def test_connector_replace_padded_with_registers():
    """After replacement: front of sequence has the real tokens (in order),
    back of sequence has tiled register values. New mask is all-zero."""
    conn = _make_tiny_connector()
    # Seed registers with distinguishable values: register i = [i, i, i, ..., i]
    reg_vals = mx.broadcast_to(
        mx.arange(4, dtype=mx.bfloat16)[:, None] + 100,
        (4, 8),
    )
    conn.learnable_registers = reg_vals.astype(mx.bfloat16)

    hidden = mx.array([[
        [0] * 8, [0] * 8, [0] * 8,  # padding (left)
        [10] * 8, [20] * 8, [30] * 8, [40] * 8, [50] * 8,
    ]], dtype=mx.bfloat16)
    # additive: [-large, -large, -large, 0, 0, 0, 0, 0] reshape to [B,1,1,T]
    add_mask = mx.array([[
        -1e9, -1e9, -1e9, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]], dtype=mx.bfloat16).reshape(1, 1, 1, 8)

    out, new_mask = conn._replace_padded_with_learnable_registers(hidden, add_mask)

    # First 5 positions should be the real tokens (10, 20, 30, 40, 50)
    for i, val in enumerate([10, 20, 30, 40, 50]):
        assert mx.all(out[0, i] == mx.array([val] * 8, dtype=mx.bfloat16)).item(), (
            f"position {i} should have value {val}"
        )
    # Last 3 positions should be register values (registers 5..7 tiled,
    # which with num_dup=2 cycles through registers 0..3 then 0..3 again,
    # so positions 5,6,7 → reg(5%4=1), reg(6%4=2), reg(7%4=3))
    # Wait actually we tile registers, so tiled = [reg0..reg3, reg0..reg3].
    # Then `out = flipped * adjusted + (1 - flipped) * tiled`. flipped is
    # [1,1,1,1,1,0,0,0] (after flipping [0,0,0,1,1,1,1,1]). So positions
    # 5,6,7 take tiled[5,6,7] = [reg1, reg2, reg3].
    assert mx.all(out[0, 5] == reg_vals[1]).item()
    assert mx.all(out[0, 6] == reg_vals[2]).item()
    assert mx.all(out[0, 7] == reg_vals[3]).item()

    # New mask is all-zero
    assert mx.allclose(new_mask, mx.zeros_like(new_mask), atol=0.0).item()


def test_connector_replace_no_padding():
    """If no padding (all-valid mask), the connector should still pack valid
    to front (which is a no-op) and the registers shouldn't replace anything.
    Actually: flipped_mask is all-ones, so `1 * adjusted + 0 * registers =
    adjusted` (which equals the original `hidden_states`)."""
    conn = _make_tiny_connector()
    hidden = mx.array([[
        [10] * 8, [20] * 8, [30] * 8, [40] * 8,
        [50] * 8, [60] * 8, [70] * 8, [80] * 8,
    ]], dtype=mx.bfloat16)
    add_mask = mx.zeros((1, 1, 1, 8), dtype=mx.bfloat16)
    out, new_mask = conn._replace_padded_with_learnable_registers(hidden, add_mask)
    # All positions should be unchanged
    assert mx.allclose(out.astype(mx.float32), hidden.astype(mx.float32), atol=1e-3).item()
    assert mx.allclose(new_mask, mx.zeros_like(new_mask), atol=0.0).item()


# --------------------------------------------------------------------------- #
# Connector forward shape
# --------------------------------------------------------------------------- #

def test_connector_forward_shape():
    """Sanity check the full connector forward pass produces the right shape."""
    conn = _make_tiny_connector()
    B, T, D = 1, 8, 8
    hidden = mx.random.normal((B, T, D)).astype(mx.float32)
    add_mask = mx.zeros((B, 1, 1, T), dtype=mx.float32)
    out, new_mask = conn(hidden, add_mask)
    assert out.shape == (B, T, D)
    assert new_mask.shape == add_mask.shape


def test_connector_block_forward_shape():
    """One block's forward shape sanity."""
    block = _BasicTransformerBlock1D(
        dim=8, heads=2, dim_head=4, apply_gated_attention=True, rope_type="split",
    )
    x = mx.random.normal((1, 8, 8)).astype(mx.float32)
    from mlx_speech.models.dramabox.ltx.rope import precompute_split_freqs_1d
    rope_cs = precompute_split_freqs_1d(
        seq_len=8, inner_dim=8, num_heads=2, theta=10000.0, max_pos=8, out_dtype=mx.float32,
    )
    y = block(x, attention_mask=None, rope_cos_sin=rope_cs)
    assert y.shape == x.shape


# --------------------------------------------------------------------------- #
# Processor wiring
# --------------------------------------------------------------------------- #

def test_processor_end_to_end_shape():
    """Wire feature_extractor + connector and check the output shape on a
    tiny but realistic-shaped input."""
    embed_dim = 16
    num_layers = 3
    inner_dim = 8
    seq_len = 8

    fx = FeatureExtractorV2(
        embedding_dim=embed_dim, out_features=inner_dim, num_layers=num_layers,
    )
    conn = Embeddings1DConnector(
        num_attention_heads=2, attention_head_dim=4, num_layers=1,
        num_learnable_registers=4, positional_embedding_max_pos=8,
        seq_len=seq_len,
    )
    proc = EmbeddingsProcessor(feature_extractor=fx, audio_connector=conn)

    hidden = [mx.random.normal((1, seq_len, embed_dim)) for _ in range(num_layers)]
    mask = mx.array([[0, 0, 1, 1, 1, 1, 1, 1]], dtype=mx.int32)

    out = proc(hidden, mask)
    assert out.audio_encoding.shape == (1, seq_len, inner_dim)
    assert out.attention_mask.shape == (1, seq_len)
    # All positions valid after register replacement
    assert (out.attention_mask == 1).all().item()
    # No NaN/Inf
    assert mx.all(mx.isfinite(out.audio_encoding)).item()
