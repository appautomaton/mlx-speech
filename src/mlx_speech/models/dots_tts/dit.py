"""Pure-MLX diffusion transformer used by both dots.tts solver variants."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import DotsTTSTransformerConfig


def _silu(value: mx.array) -> mx.array:
    return value * mx.sigmoid(value)


def _gelu_tanh(value: mx.array) -> mx.array:
    source_dtype = value.dtype
    value = value.astype(mx.float32)
    inner = math.sqrt(2.0 / math.pi) * (value + 0.044715 * value**3)
    return (0.5 * value * (1.0 + mx.tanh(inner))).astype(source_dtype)


def sinusoidal_embedding(
    timesteps: mx.array,
    dimension: int,
    *,
    max_period: float = 10_000.0,
) -> mx.array:
    """Return the official cos-first continuous-timestep embedding."""

    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must be rank one, got {timesteps.shape}")
    if dimension <= 0:
        raise ValueError("timestep embedding dimension must be positive")
    half = dimension // 2
    if half == 0:
        return mx.zeros((timesteps.shape[0], dimension), dtype=mx.float32)
    frequencies = mx.exp(
        -math.log(max_period)
        * mx.arange(half, dtype=mx.float32)
        / float(half)
    )
    arguments = timesteps.astype(mx.float32)[:, None] * frequencies[None]
    embedding = mx.concatenate((mx.cos(arguments), mx.sin(arguments)), axis=-1)
    if dimension % 2:
        embedding = mx.concatenate(
            (embedding, mx.zeros_like(embedding[:, :1])), axis=-1
        )
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = int(frequency_embedding_size)
        self.fc1 = nn.Linear(self.frequency_embedding_size, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, timesteps: mx.array) -> mx.array:
        embedding = sinusoidal_embedding(
            timesteps, self.frequency_embedding_size
        )
        return self.fc2(_silu(self.fc1(embedding)))


class AffineFreeLayerNorm(nn.Module):
    def __init__(self, dimension: int, eps: float = 1e-5):
        super().__init__()
        self.dimension = int(dimension)
        self.eps = float(eps)

    def __call__(self, value: mx.array) -> mx.array:
        source_dtype = value.dtype
        value = value.astype(mx.float32)
        mean = mx.mean(value, axis=-1, keepdims=True)
        variance = mx.mean((value - mean) ** 2, axis=-1, keepdims=True)
        return ((value - mean) * mx.rsqrt(variance + self.eps)).astype(source_dtype)


class AffineRMSNorm(nn.Module):
    """Torch-compatible affine RMSNorm used for DiT query/key normalization."""

    def __init__(self, dimension: int, eps: float = 1.1920928955078125e-7):
        super().__init__()
        self.weight = mx.ones((dimension,))
        self.eps = float(eps)

    def __call__(self, value: mx.array) -> mx.array:
        source_dtype = value.dtype
        value = value.astype(mx.float32)
        variance = mx.mean(value * value, axis=-1, keepdims=True)
        normalized = value * mx.rsqrt(variance + self.eps)
        return (normalized * self.weight.astype(mx.float32)).astype(source_dtype)


def _rotate_half(value: mx.array) -> mx.array:
    first, second = mx.split(value, 2, axis=-1)
    return mx.concatenate((-second, first), axis=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dimension: int, theta: float):
        super().__init__()
        if dimension % 2:
            raise ValueError("rotary head dimension must be even")
        self.dimension = int(dimension)
        self.theta = float(theta)

    def __call__(self, positions: mx.array) -> mx.array:
        indices = mx.arange(0, self.dimension, 2, dtype=mx.float32)
        inverse = 1.0 / (self.theta ** (indices / self.dimension))
        frequencies = positions.astype(mx.float32)[..., None] * inverse
        return mx.concatenate((frequencies, frequencies), axis=-1)

    @staticmethod
    def apply(value: mx.array, frequencies: mx.array) -> mx.array:
        source_dtype = value.dtype
        if frequencies.ndim == 2:
            frequencies = frequencies[None, None]
        elif frequencies.ndim == 3:
            frequencies = frequencies[:, None]
        else:
            raise ValueError("rotary frequencies must have rank two or three")
        value = value.astype(mx.float32)
        output = value * mx.cos(frequencies) + _rotate_half(value) * mx.sin(
            frequencies
        )
        return output.astype(source_dtype)


def _attention_bias(
    mask: mx.array | None,
    *,
    batch_size: int,
    sequence_length: int,
    dtype: mx.Dtype,
) -> mx.array | None:
    if mask is None:
        return None
    if mask.ndim == 2:
        if mask.shape == (batch_size, sequence_length):
            mask = mask[:, None, None, :]
        elif mask.shape == (sequence_length, sequence_length):
            mask = mask[None, None]
        else:
            raise ValueError(f"unsupported rank-two attention mask: {mask.shape}")
    elif mask.ndim == 3:
        if mask.shape[-2:] != (sequence_length, sequence_length):
            raise ValueError(f"attention mask sequence dimensions differ: {mask.shape}")
        mask = mask[:, None]
    elif mask.ndim != 4:
        raise ValueError(f"attention mask must have rank 2-4, got {mask.shape}")
    if mask.dtype == mx.bool_:
        return mx.where(mask, 0.0, float("-inf")).astype(dtype)
    return mask.astype(dtype)


class DiTAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        qkv_bias: bool,
        qk_norm: bool,
        norm_layer: str,
        rotary_bias: bool,
        rotary_theta: float,
    ):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("DiT hidden size must divide evenly by its heads")
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.q_norm = self._norm(norm_layer) if qk_norm else None
        self.k_norm = self._norm(norm_layer) if qk_norm else None
        self.rotary = (
            RotaryEmbedding(self.head_dim, rotary_theta) if rotary_bias else None
        )

    def _norm(self, name: str) -> nn.Module:
        if name == "RMSNorm":
            return AffineRMSNorm(self.head_dim)
        if name == "LayerNorm":
            return nn.LayerNorm(self.head_dim)
        raise ValueError(f"unsupported DiT qk norm: {name}")

    def __call__(
        self,
        value: mx.array,
        *,
        mask: mx.array | None = None,
        positions: mx.array | None = None,
    ) -> mx.array:
        batch_size, sequence_length, _ = value.shape

        def project(layer: nn.Linear) -> mx.array:
            return layer(value).reshape(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(0, 2, 1, 3)

        query, key, projected_value = (
            project(self.q_proj),
            project(self.k_proj),
            project(self.v_proj),
        )
        if self.q_norm is not None and self.k_norm is not None:
            query = self.q_norm(query)
            key = self.k_norm(key)
        if self.rotary is not None:
            if positions is None:
                positions = mx.arange(sequence_length, dtype=mx.float32)
            if positions.shape[-1] != sequence_length:
                raise ValueError("position count differs from DiT sequence length")
            frequencies = self.rotary(positions)
            query = self.rotary.apply(query, frequencies)
            key = self.rotary.apply(key, frequencies)
        bias = _attention_bias(
            mask,
            batch_size=batch_size,
            sequence_length=sequence_length,
            dtype=query.dtype,
        )
        attended = mx.fast.scaled_dot_product_attention(
            query,
            key,
            projected_value,
            scale=self.scale,
            mask=bias,
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        return self.o_proj(attended)


class DiTMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def __call__(self, value: mx.array) -> mx.array:
        return self.fc2(_gelu_tanh(self.fc1(value)))


def modulate(value: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return value * (1.0 + scale[:, None]) + shift[:, None]


class DiTBlock(nn.Module):
    def __init__(self, config: DotsTTSTransformerConfig):
        super().__init__()
        hidden_size = config.hidden_size
        self.norm1 = AffineFreeLayerNorm(hidden_size)
        self.norm2 = AffineFreeLayerNorm(hidden_size)
        self.attn = DiTAttention(
            hidden_size,
            config.num_heads,
            qkv_bias=config.qkv_bias,
            qk_norm=config.qk_norm,
            norm_layer=config.norm_layer,
            rotary_bias=config.rotary_bias,
            rotary_theta=config.rotary_theta,
        )
        self.ffn = DiTMLP(hidden_size, config.ffn_hidden_size)
        self.adaLN_modulation = nn.Linear(hidden_size, 6 * hidden_size, bias=True)

    def __call__(
        self,
        value: mx.array,
        condition: mx.array,
        *,
        mask: mx.array | None,
        positions: mx.array | None,
    ) -> mx.array:
        modulation = self.adaLN_modulation(_silu(condition))
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = (
            mx.split(modulation, 6, axis=-1)
        )
        attended = self.attn(
            modulate(self.norm1(value), shift_attn, scale_attn),
            mask=mask,
            positions=positions,
        )
        value = value + gate_attn[:, None] * attended
        feed_forward = self.ffn(
            modulate(self.norm2(value), shift_ffn, scale_ffn)
        )
        return value + gate_ffn[:, None] * feed_forward


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm = AffineFreeLayerNorm(hidden_size)
        self.adaLN_modulation = nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)

    def __call__(self, value: mx.array, condition: mx.array) -> mx.array:
        shift, scale = mx.split(
            self.adaLN_modulation(_silu(condition)), 2, axis=-1
        )
        return self.linear(modulate(self.norm(value), shift, scale))


class DiT(nn.Module):
    """Shared dots.tts DiT; MeanFlow adds the duration embedder only."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        config: DotsTTSTransformerConfig,
        *,
        meanflow: bool = False,
        frequency_embedding_size: int = 256,
    ):
        super().__init__()
        if not config.modulation:
            raise ValueError("dots.tts DiT requires adaptive modulation")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(config.hidden_size)
        self.meanflow = bool(meanflow)
        self.input_layer = nn.Linear(input_size, config.hidden_size, bias=True)
        self.time_embedder = TimestepEmbedder(
            config.hidden_size, frequency_embedding_size
        )
        self.duration_embedder = (
            TimestepEmbedder(config.hidden_size, frequency_embedding_size)
            if meanflow
            else None
        )
        self.blocks = [DiTBlock(config) for _ in range(config.num_layers)]
        self.output_layer = DiTFinalLayer(config.hidden_size, output_size)

    def __call__(
        self,
        sequence: mx.array,
        timesteps: mx.array,
        *,
        duration: mx.array | None = None,
        attention_mask: mx.array | None = None,
        positions: mx.array | None = None,
        speaker_condition: mx.array | None = None,
    ) -> mx.array:
        if sequence.ndim != 3 or int(sequence.shape[-1]) != self.input_size:
            raise ValueError(
                f"DiT expects (batch, sequence, {self.input_size}), got {sequence.shape}"
            )
        batch_size = int(sequence.shape[0])
        if timesteps.shape != (batch_size,):
            raise ValueError(f"DiT timesteps must have shape ({batch_size},)")
        condition = self.time_embedder(timesteps)
        if self.duration_embedder is not None:
            if duration is None or duration.shape != (batch_size,):
                raise ValueError(
                    f"MeanFlow duration must have shape ({batch_size},)"
                )
            condition = condition + self.duration_embedder(duration)
        if speaker_condition is not None:
            if speaker_condition.shape != (batch_size, self.hidden_size):
                raise ValueError(
                    "speaker condition must match the DiT batch and hidden size"
                )
            condition = condition + speaker_condition
        value = self.input_layer(sequence)
        for block in self.blocks:
            value = block(
                value,
                condition,
                mask=attention_mask,
                positions=positions,
            )
        return self.output_layer(value, condition)


__all__ = [
    "DiT",
    "DiTAttention",
    "DiTBlock",
    "TimestepEmbedder",
    "modulate",
    "sinusoidal_embedding",
]
