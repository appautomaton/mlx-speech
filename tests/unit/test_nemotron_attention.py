from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.nemotron_asr.attention import (
    NEG_INF,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    create_chunked_limited_mask,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "nemotron" / "attention.npz"
_CONTEXTS = ((56, 0), (56, 1), (56, 3), (56, 6), (56, 13))


def _numpy_mask(seq_len: int, left: int, right: int) -> np.ndarray:
    chunk_size = right + 1
    chunks = np.arange(seq_len, dtype=np.int32) // chunk_size
    difference = chunks[:, None] - chunks[None, :]
    visible = (difference >= 0) & (difference <= left // chunk_size)
    return np.where(visible, 0.0, NEG_INF).astype(np.float32)[None, None]


@pytest.mark.parametrize(("left", "right"), _CONTEXTS)
def test_chunked_limited_mask_matches_nemo(left: int, right: int) -> None:
    mask = create_chunked_limited_mask(73, left, right)
    mx.eval(mask)
    np.testing.assert_array_equal(np.asarray(mask), _numpy_mask(73, left, right))


def test_chunked_mask_exposes_own_and_allowed_previous_chunks() -> None:
    mask = np.asarray(create_chunked_limited_mask(12, 4, 1))[0, 0]

    assert np.all(mask[4:6, 4:6] == 0.0)
    assert np.all(mask[4:6, 0:4] == 0.0)
    assert np.all(mask[4:6, 6:] == NEG_INF)
    assert np.all(mask[10:12, :6] == NEG_INF)


def test_rel_shift_matches_hand_computed_case() -> None:
    values = mx.arange(15, dtype=mx.float32).reshape(1, 1, 3, 5)
    shifted = RelPositionMultiHeadAttention.rel_shift(values)
    mx.eval(shifted)

    expected = np.asarray([[[[2, 3, 4, 0, 5], [6, 7, 8, 9, 0], [10, 11, 12, 13, 14]]]])
    np.testing.assert_array_equal(np.asarray(shifted), expected)


def test_attention_matches_captured_torch_reference() -> None:
    with np.load(_FIXTURE) as data:
        fixture = {key: data[key] for key in data.files}

    attention = RelPositionMultiHeadAttention(2, 8, use_bias=False)
    for name in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
        getattr(attention, name).weight = mx.array(fixture[f"{name}_weight"])
    attention.pos_bias_u = mx.array(fixture["pos_bias_u"])
    attention.pos_bias_v = mx.array(fixture["pos_bias_v"])

    output = attention(
        mx.array(fixture["input"]),
        mx.array(fixture["positions"]),
        mx.array(fixture["mask"]),
    )
    mx.eval(output)
    # MLX's fused SDPA differs from torch's unfused reference by ~1e-4 here.
    np.testing.assert_allclose(
        np.asarray(output), fixture["output"], rtol=5e-3, atol=2e-4
    )


def test_projection_biases_are_disabled_and_position_biases_are_untied() -> None:
    first = RelPositionMultiHeadAttention(8, 1024, use_bias=False)
    second = RelPositionMultiHeadAttention(8, 1024, use_bias=False)

    for name in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
        assert "bias" not in getattr(first, name)
    first.pos_bias_u = mx.ones_like(first.pos_bias_u)
    assert mx.all(second.pos_bias_u == 0.0).item()
    assert first.pos_bias_u.shape == (8, 128)
    assert first.pos_bias_v.shape == (8, 128)


def test_positional_encoding_has_two_t_minus_one_positions() -> None:
    encoding = RelPositionalEncoding(8, max_len=16, scale_input=False)
    values, positions = encoding(mx.ones((1, 5, 8)))
    mx.eval(values, positions)

    assert values.shape == (1, 5, 8)
    assert positions.shape == (1, 9, 8)
    assert mx.array_equal(values, mx.ones_like(values)).item()


def test_mask_rejects_negative_right_context() -> None:
    with pytest.raises(ValueError, match="right_context"):
        create_chunked_limited_mask(8, 56, -1)
