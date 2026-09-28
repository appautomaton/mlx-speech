"""Causal Qwen-style decoder stack shared by the backbone and depth decoder.

RMSNorm multiplies by the stored weight. That is not the Gemma offset used
by the text encoder. KV entries are (batch, sequence, heads, dim).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from mlx_speech.models._cache import BoundedKVCache

from .config import BreezeDecoderConfig


def rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    dtype = x.dtype
    centered = x.astype(mx.float32)
    variance = mx.mean(mx.square(centered), axis=-1, keepdims=True)
    normalized = centered * mx.rsqrt(variance + eps)
    return (weight.astype(mx.float32) * normalized).astype(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return rms_norm(x, self.weight, self.eps)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


@dataclass(frozen=True)
class RotaryEmbedding:
    """Default or Llama 3 RoPE. Frequencies stay outside the parameter tree."""

    inv_freq: tuple[float, ...]

    @classmethod
    def from_config(cls, config: BreezeDecoderConfig) -> "RotaryEmbedding":
        head_dim = config.head_dim
        theta = config.rope_theta
        inv_freq = [
            1.0 / (theta ** (index / head_dim)) for index in range(0, head_dim, 2)
        ]
        scaling = config.rope_scaling
        if scaling is not None:
            low_wavelength = (
                scaling.original_max_position_embeddings / scaling.low_freq_factor
            )
            high_wavelength = (
                scaling.original_max_position_embeddings / scaling.high_freq_factor
            )
            scaled: list[float] = []
            for value in inv_freq:
                wavelength = 2.0 * math.pi / value
                if wavelength > low_wavelength:
                    scaled.append(value / scaling.factor)
                elif wavelength < high_wavelength:
                    scaled.append(value)
                else:
                    smooth = (
                        scaling.original_max_position_embeddings / wavelength
                        - scaling.low_freq_factor
                    ) / (scaling.high_freq_factor - scaling.low_freq_factor)
                    divided = value / scaling.factor
                    scaled.append((1.0 - smooth) * divided + smooth * value)
            inv_freq = scaled
        return cls(tuple(inv_freq))

    def embeddings(
        self, position_ids: mx.array, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        inv_freq = mx.array(self.inv_freq, dtype=mx.float32)
        freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
        angles = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def apply_rope(
    query: mx.array, key: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (
        (query * cos) + (_rotate_half(query) * sin),
        (key * cos) + (_rotate_half(key) * sin),
    )


def causal_mask(query_length: int, key_length: int, dtype: mx.Dtype) -> mx.array | None:
    """Mask queries that sit at the end of a cached key sequence."""

    if query_length == 1 and query_length == key_length:
        return None
    if query_length == 1:
        return None
    query_positions = mx.arange(key_length - query_length, key_length)
    key_positions = mx.arange(key_length)
    blocked = key_positions[None, :] > query_positions[:, None]
    minimum = mx.array(mx.finfo(dtype).min, dtype=dtype)
    zero = mx.array(0, dtype=dtype)
    return mx.where(blocked, minimum, zero)[None, None, :, :]


class DecoderAttention(nn.Module):
    def __init__(self, config: BreezeDecoderConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = float(config.head_dim) ** -0.5
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = (
            RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        )
        self.k_norm = (
            RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else None
        )

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        cache: BoundedKVCache,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(
            batch, length, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).reshape(
            batch, length, self.num_kv_heads, self.head_dim
        )
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = query.transpose(0, 2, 1, 3)
        key = key.transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)
        cos, sin = position_embeddings
        query, key = apply_rope(query, key, cos, sin)
        cache.append(key.transpose(0, 2, 1, 3), value.transpose(0, 2, 1, 3))
        cached_key = cache.keys[:, : cache.offset].transpose(0, 2, 1, 3)
        cached_value = cache.values[:, : cache.offset].transpose(0, 2, 1, 3)
        output = mx.fast.scaled_dot_product_attention(
            query,
            cached_key,
            cached_value,
            scale=self.scale,
            mask=causal_mask(length, cache.offset, hidden_states.dtype),
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, config: BreezeDecoderConfig) -> None:
        super().__init__()
        self.self_attn = DecoderAttention(config)
        self.mlp = _SwiGLU(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: tuple[mx.array, mx.array],
        cache: BoundedKVCache,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings, cache
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


class _SwiGLU(nn.Module):
    def __init__(self, config: BreezeDecoderConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def make_decoder_cache(
    config: BreezeDecoderConfig, *, batch_size: int, dtype: mx.Dtype
) -> list[BoundedKVCache]:
    return [
        BoundedKVCache.allocate(
            batch_size=batch_size,
            capacity=config.cache_growth,
            num_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            key_dtype=dtype,
            value_dtype=dtype,
            max_capacity=config.max_position_embeddings,
            growth_step=config.cache_growth,
        )
        for _ in range(config.num_hidden_layers)
    ]


def run_causal_decoder(
    layers: list[DecoderLayer],
    norm: RMSNorm,
    rope: RotaryEmbedding,
    inputs_embeds: mx.array,
    cache: list[BoundedKVCache],
    *,
    position_ids: mx.array | None = None,
) -> mx.array:
    if len(cache) != len(layers):
        raise ValueError(f"Expected {len(layers)} cache layers, got {len(cache)}.")
    batch, length, _ = inputs_embeds.shape
    offset = cache[0].offset
    if position_ids is None:
        position_ids = mx.broadcast_to(
            mx.arange(offset, offset + length)[None, :], (batch, length)
        )
    embeddings = rope.embeddings(position_ids, inputs_embeds.dtype)
    hidden_states = inputs_embeds
    for layer, layer_cache in zip(layers, cache, strict=True):
        hidden_states = layer(hidden_states, embeddings, layer_cache)
    return norm(hidden_states)
