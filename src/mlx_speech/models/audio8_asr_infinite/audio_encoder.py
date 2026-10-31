"""Incremental Voxtral audio encoder for Audio8."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import Audio8AudioConfig


class Audio8RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # Normalize in float32 and round before scaling: the reference order.
        return self.weight * mx.fast.rms_norm(hidden_states, None, self.eps)


class Audio8RotaryEmbedding(nn.Module):
    def __init__(self, config: Audio8AudioConfig) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (
            config.rope_theta
            ** (mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim)
        )

    def __call__(
        self, positions: mx.array, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        frequencies = positions.astype(mx.float32)[:, None] * self.inv_freq[None, :]
        angles = mx.concatenate([frequencies, frequencies], axis=-1)
        return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def apply_rotary(
    values: mx.array, position_embeddings: tuple[mx.array, mx.array]
) -> mx.array:
    cos, sin = position_embeddings
    return values * cos + _rotate_half(values) * sin


@dataclass
class Audio8LayerCache:
    capacity: int = 750
    _keys: mx.array | None = None
    _values: mx.array | None = None
    _positions: mx.array | None = None
    write_offset: int = 0
    length: int = 0

    @property
    def keys(self) -> mx.array | None:
        return self._logical(self._keys)

    @property
    def values(self) -> mx.array | None:
        return self._logical(self._values)

    @property
    def positions(self) -> mx.array | None:
        if self._positions is None:
            return None
        if self.length < self.capacity:
            return self._positions[: self.length]
        return self._positions[self.write_offset : self.write_offset + self.capacity]

    def _logical(self, buffer: mx.array | None) -> mx.array | None:
        if buffer is None:
            return None
        if self.length < self.capacity:
            return buffer[:, :, : self.length]
        return buffer[:, :, self.write_offset : self.write_offset + self.capacity]

    def append(
        self, keys: mx.array, values: mx.array, positions: mx.array
    ) -> tuple[mx.array, mx.array, mx.array]:
        count = keys.shape[2]
        if count == 0:
            return self.keys, self.values, self.positions
        if count >= self.capacity:
            self.replace(
                keys[:, :, -self.capacity :],
                values[:, :, -self.capacity :],
                positions[-self.capacity :],
            )
            return self.keys, self.values, self.positions
        if self._keys is None:
            self._allocate(keys, values, positions)
        first = min(count, self.capacity - self.write_offset)
        self._write(self.write_offset, keys[:, :, :first], values[:, :, :first], positions[:first])
        remaining = count - first
        if remaining:
            self._write(0, keys[:, :, first:], values[:, :, first:], positions[first:])
        self.write_offset = (self.write_offset + count) % self.capacity
        self.length = min(self.capacity, self.length + count)
        return self.keys, self.values, self.positions

    def replace(
        self, keys: mx.array, values: mx.array, positions: mx.array
    ) -> None:
        self._allocate(keys, values, positions)
        self._write(0, keys, values, positions)
        self.length = keys.shape[2]
        self.write_offset = self.length % self.capacity

    def _allocate(
        self, keys: mx.array, values: mx.array, positions: mx.array
    ) -> None:
        batch, heads, _, width = keys.shape
        self._keys = mx.zeros((batch, heads, self.capacity * 2, width), dtype=keys.dtype)
        self._values = mx.zeros((batch, heads, self.capacity * 2, width), dtype=values.dtype)
        self._positions = mx.zeros((self.capacity * 2,), dtype=positions.dtype)

    def _write(
        self,
        start: int,
        keys: mx.array,
        values: mx.array,
        positions: mx.array,
    ) -> None:
        count = keys.shape[2]
        end = start + count
        self._keys[:, :, start:end] = keys
        self._keys[:, :, start + self.capacity : end + self.capacity] = keys
        self._values[:, :, start:end] = values
        self._values[:, :, start + self.capacity : end + self.capacity] = values
        self._positions[start:end] = positions
        self._positions[start + self.capacity : end + self.capacity] = positions


@dataclass
class Audio8AudioCache:
    layers: list[Audio8LayerCache]
    position_offset: int = 0


class Audio8CausalConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__(in_channels, out_channels, 3, stride=stride)
        self.left_pad = 3 - stride

    def __call__(self, hidden_states: mx.array) -> mx.array:
        padded = mx.pad(hidden_states, ((0, 0), (self.left_pad, 0), (0, 0)))
        return super().__call__(padded)


class Audio8Embedder(nn.Module):
    def __init__(self, config: Audio8AudioConfig) -> None:
        super().__init__()
        self.conv1 = Audio8CausalConv1d(config.num_mel_bins, config.hidden_size, stride=1)
        self.conv2 = Audio8CausalConv1d(config.hidden_size, config.hidden_size, stride=2)

    def __call__(self, input_features: mx.array) -> mx.array:
        hidden_states = input_features.transpose(0, 2, 1).astype(self.conv1.weight.dtype)
        hidden_states = nn.gelu(self.conv1(hidden_states))
        return nn.gelu(self.conv2(hidden_states))


class Audio8Attention(nn.Module):
    def __init__(self, config: Audio8AudioConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.output_size = self.num_heads * self.head_dim
        self.scale = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(config.hidden_size, self.output_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.output_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.output_size, bias=True)
        self.o_proj = nn.Linear(self.output_size, config.hidden_size, bias=True)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        position_embeddings: tuple[mx.array, mx.array],
        positions: mx.array,
        cache: Audio8LayerCache | None,
    ) -> mx.array:
        batch, sequence, _ = hidden_states.shape
        query = self.q_proj(hidden_states).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key = self.k_proj(hidden_states).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value = self.v_proj(hidden_states).reshape(
            batch, sequence, self.num_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        query = apply_rotary(query, position_embeddings)
        key = apply_rotary(key, position_embeddings)
        if cache is None:
            key_positions = positions
        else:
            key, value, key_positions = cache.append(key, value, positions)
        allowed = mx.logical_and(
            key_positions[None, :] <= positions[:, None],
            key_positions[None, :] > positions[:, None] - self.sliding_window,
        )
        attended = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=allowed[None, None, :, :]
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, sequence, self.output_size)
        return self.o_proj(attended)


class Audio8MLP(nn.Module):
    def __init__(self, config: Audio8AudioConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class Audio8EncoderLayer(nn.Module):
    def __init__(self, config: Audio8AudioConfig) -> None:
        super().__init__()
        self.self_attn = Audio8Attention(config)
        self.self_attn_layer_norm = Audio8RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.final_layer_norm = Audio8RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Audio8MLP(config)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        position_embeddings: tuple[mx.array, mx.array],
        positions: mx.array,
        cache: Audio8LayerCache | None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.self_attn(
            self.self_attn_layer_norm(hidden_states),
            position_embeddings=position_embeddings,
            positions=positions,
            cache=cache,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.final_layer_norm(hidden_states))
        return residual + hidden_states


@dataclass(frozen=True)
class Audio8EncoderOutput:
    last_hidden_state: mx.array


class Audio8Encoder(nn.Module):
    def __init__(self, config: Audio8AudioConfig) -> None:
        super().__init__()
        self.config = config
        self.embedder = Audio8Embedder(config)
        self.layers = [Audio8EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = Audio8RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Audio8RotaryEmbedding(config)

    def new_cache(self, *, max_query_frames: int = 8) -> Audio8AudioCache:
        if max_query_frames <= 0:
            raise ValueError("Audio8 max_query_frames must be positive")
        capacity = self.config.sliding_window + max_query_frames - 1
        return Audio8AudioCache(
            layers=[
                Audio8LayerCache(capacity=capacity)
                for _ in range(self.config.num_hidden_layers)
            ]
        )

    def __call__(
        self,
        *,
        input_features: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: Audio8AudioCache | None = None,
        positions: mx.array | None = None,
    ) -> Audio8EncoderOutput:
        if (input_features is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_features or inputs_embeds")
        hidden_states = (
            self.embedder(input_features) if inputs_embeds is None else inputs_embeds
        )
        sequence = hidden_states.shape[1]
        offset = cache.position_offset if cache is not None else 0
        if positions is None:
            positions = mx.arange(offset, offset + sequence, dtype=mx.int32)
        elif positions.ndim == 2 and positions.shape[0] == 1:
            positions = positions[0]
        if positions.shape != (sequence,):
            raise ValueError(f"Expected positions [{sequence}], got {positions.shape}")
        position_embeddings = self.rotary_emb(positions, hidden_states.dtype)
        for index, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                positions=positions,
                cache=cache.layers[index] if cache is not None else None,
            )
        hidden_states = self.norm(hidden_states)
        if cache is not None:
            cache.position_offset = offset + sequence
            arrays = [hidden_states]
            for layer in cache.layers:
                arrays.extend((layer._keys, layer._values, layer._positions))
            mx.eval(*arrays)
        return Audio8EncoderOutput(last_hidden_state=hidden_states)


def _rotate_half(values: mx.array) -> mx.array:
    half = values.shape[-1] // 2
    return mx.concatenate([-values[..., half:], values[..., :half]], axis=-1)
