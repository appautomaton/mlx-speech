"""Breeze T5Gemma2 text encoder.

Full-attention layers are bidirectional. Sliding layers use the official
local window, which also looks ahead. Neither mask is causal.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import BreezeTextEncoderConfig


def gemma_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """RMSNorm whose checkpoint weight is stored as an offset from one."""

    dtype = x.dtype
    centered = x.astype(mx.float32)
    variance = mx.mean(mx.square(centered), axis=-1, keepdims=True)
    normalized = centered * mx.rsqrt(variance + eps)
    output = normalized * (1.0 + weight.astype(mx.float32))
    return output.astype(dtype)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        return gemma_rms_norm(x, self.weight, self.eps)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


class TextRotaryEmbedding:
    """Half-split RoPE. A linear factor divides the inverse frequencies."""

    def __init__(
        self, head_dim: int, *, base: float, linear_factor: float | None
    ) -> None:
        inv_freq = [
            1.0 / (float(base) ** (index / head_dim)) for index in range(0, head_dim, 2)
        ]
        if linear_factor is not None:
            inv_freq = [value / float(linear_factor) for value in inv_freq]
        self._inv_freq = tuple(inv_freq)

    def embeddings(
        self, position_ids: mx.array, dtype: mx.Dtype
    ) -> tuple[mx.array, mx.array]:
        inv_freq = mx.array(self._inv_freq, dtype=mx.float32)
        freqs = position_ids.astype(mx.float32)[..., None] * inv_freq
        angles = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(angles).astype(dtype), mx.sin(angles).astype(dtype)


def apply_text_rope(
    query: mx.array,
    key: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    rotated_query = (query * cos) + (_rotate_half(query) * sin)
    rotated_key = (key * cos) + (_rotate_half(key) * sin)
    return rotated_query, rotated_key


def text_attention_mask(
    attention_mask: mx.array | None,
    *,
    batch_size: int,
    sequence_length: int,
    layer_type: str,
    sliding_window: int,
    dtype: mx.Dtype,
) -> mx.array | None:
    """Return an additive mask, or None when every key is visible."""

    if layer_type not in {"full_attention", "sliding_attention"}:
        raise ValueError(f"Unsupported text attention type {layer_type!r}.")
    if attention_mask is None:
        valid = mx.ones((batch_size, sequence_length), dtype=mx.bool_)
        has_padding = False
    else:
        if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (
            batch_size,
            sequence_length,
        ):
            raise ValueError(
                "Text attention_mask must have shape "
                f"({batch_size}, {sequence_length}), got {tuple(attention_mask.shape)}."
            )
        valid = attention_mask.astype(mx.bool_)
        has_padding = not bool(mx.all(valid).item())
    if layer_type == "full_attention" and not has_padding:
        return None

    allowed = valid[:, None, None, :]
    if layer_type == "sliding_attention":
        if sliding_window <= 0:
            raise ValueError("Text sliding_window must be positive.")
        positions = mx.arange(sequence_length)
        distance = positions[:, None] - positions[None, :]
        left = (sliding_window + 1) // 2
        right = sliding_window // 2 + 1
        local = mx.logical_or(
            mx.logical_and(distance >= 0, distance < left),
            mx.logical_and(distance < 0, -distance < right),
        )
        allowed = mx.logical_and(allowed, local[None, None, :, :])
    blocked = mx.logical_not(allowed)
    minimum = mx.array(mx.finfo(dtype).min, dtype=dtype)
    zero = mx.array(0, dtype=dtype)
    return mx.where(blocked, minimum, zero)


class ScaledWordEmbedding(nn.Module):
    def __init__(self, config: BreezeTextEncoderConfig) -> None:
        super().__init__()
        self.weight = mx.zeros((config.vocab_size, config.hidden_size))
        self.eoi_embedding = mx.zeros((config.hidden_size,))
        self.scale = float(config.hidden_size) ** 0.5
        self.eoi_token_index = int(config.eoi_token_index)

    def __call__(self, input_ids: mx.array) -> mx.array:
        embedded = self.weight[input_ids] * self.scale
        eoi = self.eoi_embedding.astype(embedded.dtype)
        return mx.where((input_ids == self.eoi_token_index)[..., None], eoi, embedded)


class TextMLP(nn.Module):
    def __init__(self, config: BreezeTextEncoderConfig) -> None:
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
        return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class TextAttention(nn.Module):
    def __init__(self, config: BreezeTextEncoderConfig, layer_type: str) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = float(config.query_pre_attn_scalar) ** -0.5
        bias = config.attention_bias
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=bias
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, config.rms_norm_eps)
        if layer_type == "full_attention":
            base, factor = config.full_rope_theta, config.full_rope_factor
        elif layer_type == "sliding_attention":
            base, factor = config.sliding_rope_theta, None
        else:
            raise ValueError(f"Unsupported text attention type {layer_type!r}.")
        self.rope = TextRotaryEmbedding(self.head_dim, base=base, linear_factor=factor)
        self.layer_type = layer_type

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None,
    ) -> mx.array:
        batch, length, _ = hidden_states.shape
        query = self.q_norm(
            self.q_proj(hidden_states).reshape(
                batch, length, self.num_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        key = self.k_norm(
            self.k_proj(hidden_states).reshape(
                batch, length, self.num_kv_heads, self.head_dim
            )
        ).transpose(0, 2, 1, 3)
        value = (
            self.v_proj(hidden_states)
            .reshape(batch, length, self.num_kv_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        cos, sin = position_embeddings
        query, key = apply_text_rope(query, key, cos, sin)
        output = mx.fast.scaled_dot_product_attention(
            query, key, value, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class TextEncoderLayer(nn.Module):
    def __init__(self, config: BreezeTextEncoderConfig, layer_type: str) -> None:
        super().__init__()
        self.attention_type = layer_type
        self.self_attn = TextAttention(config, layer_type)
        self.pre_self_attn_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_self_attn_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = TextMLP(config)
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        position_embeddings: tuple[mx.array, mx.array],
        mask: mx.array | None,
    ) -> mx.array:
        residual = hidden_states
        attended = self.self_attn(
            self.pre_self_attn_layernorm(hidden_states),
            position_embeddings=position_embeddings,
            mask=mask,
        )
        hidden_states = residual + self.post_self_attn_layernorm(attended)
        residual = hidden_states
        feed_forward = self.mlp(self.pre_feedforward_layernorm(hidden_states))
        return residual + self.post_feedforward_layernorm(feed_forward)


class BreezeTextEncoder(nn.Module):
    def __init__(self, config: BreezeTextEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = ScaledWordEmbedding(config)
        self.layers = [
            TextEncoderLayer(config, layer_type) for layer_type in config.layer_types
        ]
        self.norm = GemmaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array,
        *,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        if input_ids.ndim != 2:
            raise ValueError(
                f"Text encoder input_ids must have shape (batch, sequence), got {input_ids.shape}."
            )
        hidden_states = self.embed_tokens(input_ids)
        batch, length, _ = hidden_states.shape
        if position_ids is None:
            position_ids = mx.broadcast_to(mx.arange(length)[None, :], (batch, length))
        elif position_ids.ndim == 1:
            position_ids = mx.broadcast_to(position_ids[None, :], (batch, length))
        masks: dict[str, mx.array | None] = {}
        embeddings: dict[str, tuple[mx.array, mx.array]] = {}
        for layer in self.layers:
            kind = layer.attention_type
            if kind not in masks:
                masks[kind] = text_attention_mask(
                    attention_mask,
                    batch_size=batch,
                    sequence_length=length,
                    layer_type=kind,
                    sliding_window=self.config.sliding_window,
                    dtype=hidden_states.dtype,
                )
                embeddings[kind] = layer.self_attn.rope.embeddings(
                    position_ids, hidden_states.dtype
                )
            hidden_states = layer(
                hidden_states,
                position_embeddings=embeddings[kind],
                mask=masks[kind],
            )
        return self.norm(hidden_states)


def expected_text_checkpoint_weights(
    config: BreezeTextEncoderConfig,
) -> dict[str, tuple[int, ...]]:
    """Checkpoint keys and shapes for the text encoder and its projection."""

    hidden = config.hidden_size
    intermediate = config.intermediate_size
    query = config.num_attention_heads * config.head_dim
    key = config.num_key_value_heads * config.head_dim
    bias = (hidden,) if config.attention_bias else ()
    weights: dict[str, tuple[int, ...]] = {
        "text_encoder.embed_tokens.weight": (config.vocab_size, hidden),
        "text_encoder.embed_tokens.eoi_embedding": (hidden,),
        "text_encoder.norm.weight": (hidden,),
        "text_encoder_proj.weight": (config.backbone_hidden_size, hidden),
    }
    for index in range(config.num_hidden_layers):
        prefix = f"text_encoder.layers.{index}"
        weights.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (query, hidden),
                f"{prefix}.self_attn.k_proj.weight": (key, hidden),
                f"{prefix}.self_attn.v_proj.weight": (key, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, query),
                f"{prefix}.self_attn.q_norm.weight": (config.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (config.head_dim,),
                f"{prefix}.pre_self_attn_layernorm.weight": (hidden,),
                f"{prefix}.post_self_attn_layernorm.weight": (hidden,),
                f"{prefix}.pre_feedforward_layernorm.weight": (hidden,),
                f"{prefix}.post_feedforward_layernorm.weight": (hidden,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
        if bias:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                width = (
                    hidden if name == "o_proj" else (query if name == "q_proj" else key)
                )
                weights[f"{prefix}.self_attn.{name}.bias"] = (width,)
    return weights
