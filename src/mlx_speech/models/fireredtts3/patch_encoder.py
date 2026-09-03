"""MLX latent patch encoder used by FireRedTTS3 Base."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def _linear(layer: nn.Linear, value: mx.array) -> mx.array:
    output = mx.matmul(value.astype(mx.float32), layer.weight.astype(mx.float32).T)
    if layer.bias is not None:
        output = output + layer.bias.astype(mx.float32)
    return output.astype(value.dtype)


class RotaryEmbedding:
    def __init__(self, dim: int, *, base: float = 10_000.0):
        if dim % 2:
            raise ValueError("rotary embedding dimension must be even")
        self.dim = int(dim)
        self.base = float(base)

    def frequencies(self, seq_len: int) -> mx.array:
        inv_freq = 1.0 / (
            self.base
            ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / float(self.dim))
        )
        positions = mx.arange(seq_len, dtype=mx.float32)[None, :, None]
        freqs = positions * inv_freq[None, None]
        return mx.stack((freqs, freqs), axis=-1).reshape(1, seq_len, self.dim)

    @staticmethod
    def apply(value: mx.array, frequencies: mx.array) -> mx.array:
        rotated = value.reshape(*value.shape[:-1], -1, 2)
        first = rotated[..., 0]
        second = rotated[..., 1]
        rotate_half = mx.stack((-second, first), axis=-1).reshape(value.shape)
        frequencies = frequencies[:, None, -value.shape[-2] :]
        return (
            value * mx.cos(frequencies) + rotate_half * mx.sin(frequencies)
        ).astype(value.dtype)


class _RMSNorm(nn.Module):
    def __init__(self, dim: int, *, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = float(eps)

    def __call__(self, value: mx.array) -> mx.array:
        dtype = value.dtype
        value = value.astype(mx.float32)
        value = value * mx.rsqrt(mx.mean(value * value, axis=-1, keepdims=True) + self.eps)
        return (value * self.weight.astype(mx.float32)).astype(dtype)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, *, mult: float, dropout: float):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = [
            [nn.Linear(dim, inner_dim)],
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        ]

    def __call__(self, value: mx.array) -> mx.array:
        value = nn.gelu_approx(_linear(self.ff[0][0], value))
        return _linear(self.ff[2], self.ff[1](value))


class _Attention(nn.Module):
    def __init__(self, dim: int, *, heads: int, dropout: float):
        super().__init__()
        if dim % heads:
            raise ValueError("attention dimension must be divisible by heads")
        self.heads = int(heads)
        self.head_dim = dim // heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = [nn.Linear(dim, dim), nn.Dropout(dropout)]

    def __call__(
        self,
        value: mx.array,
        *,
        rope: tuple[RotaryEmbedding, mx.array] | None,
    ) -> mx.array:
        batch, length, dim = value.shape
        query = _linear(self.to_q, value).reshape(
            batch, length, self.heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key = _linear(self.to_k, value).reshape(
            batch, length, self.heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        values = _linear(self.to_v, value).reshape(
            batch, length, self.heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        if rope is not None:
            rotary, frequencies = rope
            query = rotary.apply(query, frequencies)
            key = rotary.apply(key, frequencies)
        scores = mx.matmul(
            query.astype(mx.float32),
            key.astype(mx.float32).transpose(0, 1, 3, 2),
        ) * self.scale
        weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(weights, values.astype(mx.float32))
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, dim)
        return self.to_out[1](_linear(self.to_out[0], output.astype(value.dtype)))


class _PatchBlock(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size)
        self.attn = _Attention(hidden_size, heads=num_heads, dropout=0.1)
        self.norm2 = _RMSNorm(hidden_size)
        self.mlp = _FeedForward(hidden_size, mult=mlp_ratio, dropout=0.1)

    def __call__(
        self,
        value: mx.array,
        *,
        rope: tuple[RotaryEmbedding, mx.array],
    ) -> mx.array:
        value = value + self.attn(self.norm1(value), rope=rope)
        return value + self.mlp(self.norm2(value))


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm_final = _RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_dim)

    def __call__(self, value: mx.array) -> mx.array:
        return _linear(self.linear, self.norm_final(value))


class PatchEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        patch_size: int,
        hidden_size: int,
        mlp_ratio: float,
        depth: int,
        num_heads: int,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.patch_size = int(patch_size)
        self.hidden_size = int(hidden_size)
        self.cls_tok = mx.zeros((1, 1, hidden_size))
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = [
            _PatchBlock(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ]
        self.in_proj = nn.Linear(in_dim, hidden_size)
        self.out_proj = _FinalLayer(hidden_size, out_dim)

    def __call__(self, inputs_embeds: mx.array) -> mx.array:
        if inputs_embeds.ndim != 3 or int(inputs_embeds.shape[0]) != 1:
            raise ValueError("FireRedTTS3 PatchEncoder expects shape (1, frames, dim)")
        if int(inputs_embeds.shape[-1]) != self.in_dim:
            raise ValueError(
                f"PatchEncoder expected latent dim {self.in_dim}, "
                f"got {inputs_embeds.shape[-1]}"
            )
        if int(inputs_embeds.shape[1]) % self.patch_size:
            raise ValueError("PatchEncoder frames must be divisible by patch_size")
        value = _linear(self.in_proj, inputs_embeds)
        value = value.reshape(-1, self.patch_size, self.hidden_size)
        cls = mx.broadcast_to(self.cls_tok, (value.shape[0], 1, self.hidden_size))
        value = mx.concatenate((cls, value), axis=1)
        rope = (self.rotary_embed, self.rotary_embed.frequencies(int(value.shape[1])))
        for block in self.blocks:
            value = block(value, rope=rope)
        return self.out_proj(value)[:, 0][None]


__all__ = ["PatchEncoder", "RotaryEmbedding"]
