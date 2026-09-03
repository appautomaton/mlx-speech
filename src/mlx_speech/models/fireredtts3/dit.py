"""MLX flow-matching DiT used by FireRedTTS3 Base."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .patch_encoder import (
    RotaryEmbedding,
    _Attention,
    _FeedForward,
    _linear,
    _RMSNorm,
)


def _layer_norm(value: mx.array, *, eps: float = 1e-6) -> mx.array:
    dtype = value.dtype
    value = value.astype(mx.float32)
    mean = mx.mean(value, axis=-1, keepdims=True)
    variance = mx.mean((value - mean) ** 2, axis=-1, keepdims=True)
    return ((value - mean) * mx.rsqrt(variance + eps)).astype(dtype)


def _modulate(value: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return value * (1.0 + scale) + shift


class _TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, *, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = int(frequency_dim)
        self.time_mlp = [
            nn.Linear(frequency_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        ]

    def sinusoidal_embedding(self, timestep: mx.array) -> mx.array:
        half = self.frequency_dim // 2
        exponent = math.log(10_000.0) / (half - 1)
        frequencies = mx.exp(mx.arange(half, dtype=mx.float32) * -exponent)
        angles = 1_000.0 * timestep.astype(mx.float32)[:, None] * frequencies[None]
        return mx.concatenate((mx.sin(angles), mx.cos(angles)), axis=-1)

    def __call__(self, timestep: mx.array) -> mx.array:
        value = self.sinusoidal_embedding(timestep).astype(timestep.dtype)
        value = _linear(self.time_mlp[0], value)
        value = self.time_mlp[1](value)
        return _linear(self.time_mlp[2], value)


class _Conv1d(nn.Module):
    def __init__(self, channels: int, *, kernel_size: int = 3):
        super().__init__()
        scale = math.sqrt(1.0 / (channels * kernel_size))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(channels, kernel_size, channels),
        )
        self.bias = mx.zeros((channels,))
        self.padding = (kernel_size - 1) // 2

    def __call__(self, value: mx.array) -> mx.array:
        dtype = value.dtype
        output = mx.conv1d(
            value.astype(mx.float32),
            self.weight.astype(mx.float32),
            padding=self.padding,
        )
        return (output + self.bias.astype(mx.float32)).astype(dtype)


class _ConvBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = [_Conv1d(channels), nn.Mish(), _Conv1d(channels)]

    def __call__(self, value: mx.array) -> mx.array:
        value = self.block[0](value)
        value = self.block[1](value)
        return self.block[2](value)


class _DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = _RMSNorm(hidden_size)
        self.attn = _Attention(hidden_size, heads=num_heads, dropout=0.0)
        self.norm2 = _RMSNorm(hidden_size)
        self.conv = _ConvBlock(hidden_size)
        self.norm3 = _RMSNorm(hidden_size)
        self.mlp = _FeedForward(hidden_size, mult=mlp_ratio, dropout=0.0)
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size),
        ]

    def __call__(
        self,
        value: mx.array,
        condition: mx.array,
        *,
        rope: tuple[RotaryEmbedding, mx.array],
    ) -> mx.array:
        modulation = _linear(
            self.adaLN_modulation[1],
            self.adaLN_modulation[0](condition),
        )
        (
            shift_attention,
            scale_attention,
            gate_attention,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            shift_conv,
            scale_conv,
            gate_conv,
        ) = mx.split(modulation, 9, axis=-1)
        value = value + gate_attention * self.attn(
            _modulate(self.norm1(value), shift_attention, scale_attention),
            rope=rope,
        )
        value = value + gate_conv * self.conv(
            _modulate(self.norm2(value), shift_conv, scale_conv)
        )
        return value + gate_mlp * self.mlp(
            _modulate(self.norm3(value), shift_mlp, scale_mlp)
        )


class _FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size),
        ]
        self.linear = nn.Linear(hidden_size, out_channels)

    def __call__(self, value: mx.array, condition: mx.array) -> mx.array:
        modulation = _linear(
            self.adaLN_modulation[1],
            self.adaLN_modulation[0](condition),
        )
        shift, scale = mx.split(modulation, 2, axis=-1)
        return _linear(self.linear, _modulate(_layer_norm(value), shift, scale))


class DiT(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        mlp_ratio: float,
        depth: int,
        num_heads: int,
        hidden_size: int,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.in_proj = nn.Linear(in_channels, hidden_size)
        self.t_embedder = _TimestepEmbedder(hidden_size)
        self.rotary_embed = RotaryEmbedding(hidden_size // num_heads)
        self.blocks = [
            _DiTBlock(hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ]
        self.final_layer = _FinalLayer(hidden_size, out_channels)

    def __call__(self, value: mx.array, timestep: mx.array) -> mx.array:
        if value.ndim != 3 or int(value.shape[-1]) != self.in_channels:
            raise ValueError(
                f"FireRedTTS3 DiT expected (*, *, {self.in_channels}), "
                f"got {value.shape}"
            )
        condition = self.t_embedder(timestep.reshape(-1))[:, None]
        value = _linear(self.in_proj, value)
        rope = (self.rotary_embed, self.rotary_embed.frequencies(int(value.shape[1])))
        for block in self.blocks:
            value = block(value, condition, rope=rope)
        return self.final_layer(value, condition)


__all__ = ["DiT"]
