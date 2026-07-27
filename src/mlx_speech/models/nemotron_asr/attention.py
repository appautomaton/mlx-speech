"""Relative-position attention and cache-aware masks for Nemotron ASR."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

NEG_INF = -1e30


def create_chunked_limited_mask(
    seq_len: int, left_context: int, right_context: int
) -> mx.array:
    """Return NeMo's additive ``chunked_limited`` mask as ``[1, 1, T, T]``."""
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if right_context < 0:
        raise ValueError("right_context must be non-negative")

    chunk_size = right_context + 1
    left_chunks = left_context // chunk_size if left_context >= 0 else 10_000
    chunk_index = mx.arange(seq_len, dtype=mx.int32) // chunk_size
    difference = chunk_index[:, None] - chunk_index[None, :]
    visible = (difference >= 0) & (difference <= left_chunks)
    mask = mx.where(visible, 0.0, NEG_INF).astype(mx.float32)
    return mask[None, None, :, :]


class RelPositionalEncoding(nn.Module):
    """Transformer-XL sinusoidal positions spanning ``2T-1`` locations."""

    def __init__(
        self, d_model: int, *, max_len: int = 5000, scale_input: bool = False
    ) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even")
        if max_len < 1:
            raise ValueError("max_len must be positive")
        self.d_model = d_model
        self.max_len = max_len
        self.scale = math.sqrt(d_model) if scale_input else 1.0
        self._calculate()

    def _calculate(self) -> None:
        positions = mx.arange(
            self.max_len - 1, -self.max_len, -1, dtype=mx.float32
        )[:, None]
        divisor = mx.exp(
            mx.arange(0, self.d_model, 2, dtype=mx.float32)
            * -(math.log(10_000.0) / self.d_model)
        )
        angles = positions * divisor[None, :]
        encoding = mx.stack([mx.sin(angles), mx.cos(angles)], axis=-1).reshape(
            2 * self.max_len - 1, self.d_model
        )
        self._pe = encoding[None, :, :]
        mx.eval(self._pe)

    def _ensure_length(self, length: int) -> None:
        if length > self.max_len:
            self.max_len = length + 1
            self._calculate()

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        length = x.shape[1]
        self._ensure_length(length)
        center = self._pe.shape[1] // 2
        positions = self._pe[:, center - (length - 1) : center + length]
        return x * self.scale, positions.astype(x.dtype)

    def for_length(self, length: int, dtype: mx.Dtype = mx.float32) -> mx.array:
        """Return ``2*length-1`` positions for a cached attention window."""
        if length < 1:
            raise ValueError("length must be positive")
        self._ensure_length(length)
        center = self._pe.shape[1] // 2
        return self._pe[:, center - (length - 1) : center + length].astype(dtype)


class RelPositionMultiHeadAttention(nn.Module):
    """NeMo Transformer-XL attention with untied per-layer position biases."""

    def __init__(self, n_heads: int, d_model: int, *, use_bias: bool = False) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5

        self.linear_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_out = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = mx.zeros((n_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((n_heads, self.head_dim))

    @staticmethod
    def rel_shift(x: mx.array) -> mx.array:
        """Perform NeMo's relative-position shift on ``[B, H, T, 2T-1]``."""
        batch, heads, query_length, position_length = x.shape
        shifted = mx.pad(x, ((0, 0), (0, 0), (0, 0), (1, 0)))
        shifted = shifted.reshape(batch, heads, position_length + 1, query_length)
        shifted = shifted[:, :, 1:, :]
        return shifted.reshape(batch, heads, query_length, position_length)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        return self._attention(x, x, pos_emb, mask)

    def stream(self, query: mx.array, key_value: mx.array, pos_emb: mx.array) -> mx.array:
        """Attend a new chunk to its fixed cached window without a mask."""
        return self._attention(query, key_value, pos_emb, None)

    def _attention(
        self,
        query_input: mx.array,
        key_value_input: mx.array,
        pos_emb: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        query = self.linear_q(query_input)
        key = self.linear_k(key_value_input)
        value = self.linear_v(key_value_input)
        position = self.linear_pos(pos_emb)

        batch, query_length, _ = query.shape
        key_length = key.shape[1]
        position_length = position.shape[1]
        query = query.reshape(batch, query_length, self.n_heads, self.head_dim)
        query_u = mx.transpose(query + self.pos_bias_u, (0, 2, 1, 3))
        query_v = mx.transpose(query + self.pos_bias_v, (0, 2, 1, 3))
        key = mx.transpose(
            key.reshape(batch, key_length, self.n_heads, self.head_dim),
            (0, 2, 1, 3),
        )
        value = mx.transpose(
            value.reshape(batch, key_length, self.n_heads, self.head_dim),
            (0, 2, 1, 3),
        )
        position = mx.transpose(
            position.reshape(
                position.shape[0], position_length, self.n_heads, self.head_dim
            ),
            (0, 2, 1, 3),
        )

        positional_scores = query_v @ mx.swapaxes(position, -2, -1)
        positional_scores = self.rel_shift(positional_scores)[..., :key_length]
        additive_mask = positional_scores * self.scale
        if mask is not None:
            additive_mask = additive_mask + mask.astype(additive_mask.dtype)

        output = mx.fast.scaled_dot_product_attention(
            query_u,
            key,
            value,
            scale=self.scale,
            mask=additive_mask,
        )
        output = mx.transpose(output, (0, 2, 1, 3)).reshape(
            batch, query_length, self.d_model
        )
        return self.linear_out(output)


__all__ = [
    "NEG_INF",
    "RelPositionalEncoding",
    "RelPositionMultiHeadAttention",
    "create_chunked_limited_mask",
]
