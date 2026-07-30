"""Family-neutral pure-MLX Qwen2 decoder trunk."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol

import mlx.core as mx
import mlx.nn as nn


class Qwen2Config(Protocol):
    """Configuration fields consumed by the shared Qwen2 implementation."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    attention_dropout: float
    hidden_act: str
    model_type: str
    tie_word_embeddings: bool
    rope_theta: float


Qwen2LayerCache = tuple[mx.array, mx.array]
Qwen2KVCache = list[Qwen2LayerCache]
Qwen2RotaryDtypePolicy = Literal["float32", "query"]


class Qwen2RMSNorm(nn.Module):
    """Qwen2 RMSNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class Qwen2RotaryEmbedding(nn.Module):
    """Qwen2 rotary positions with an explicit decode offset."""

    def __init__(self, dim: int, base: float = 1_000_000.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"Qwen2 RoPE dimension must be even, got {dim}.")
        self._dim = dim
        self._base = float(base)

    def _inv_freq(self) -> mx.array:
        exponent = mx.arange(0, self._dim, 2, dtype=mx.float32) / self._dim
        return 1.0 / (self._base**exponent)

    def __call__(
        self,
        offset: int,
        seq_len: int,
        *,
        dtype: mx.Dtype = mx.float32,
    ) -> tuple[mx.array, mx.array]:
        if offset < 0:
            raise ValueError(f"Qwen2 RoPE offset must be non-negative, got {offset}.")
        if seq_len <= 0:
            raise ValueError(f"Qwen2 RoPE sequence length must be positive, got {seq_len}.")
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, self._inv_freq())
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    # q/k: (B, L, H, D), cos/sin: (L, D)
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    return (
        (q * cos) + (_rotate_half(q) * sin),
        (k * cos) + (_rotate_half(k) * sin),
    )


class Qwen2Attention(nn.Module):
    """Qwen2 grouped-query self-attention with append-only KV caching."""

    def __init__(
        self,
        config: Qwen2Config,
        *,
        rotary_dtype_policy: Qwen2RotaryDtypePolicy = "float32",
    ):
        super().__init__()
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("Qwen2 hidden size must divide evenly by attention heads.")
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Qwen2 attention heads must divide evenly by KV heads.")
        if rotary_dtype_policy not in ("float32", "query"):
            raise ValueError(
                "Qwen2 rotary dtype policy must be 'float32' or 'query', "
                f"got {rotary_dtype_policy!r}."
            )

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rotary_dtype_policy = rotary_dtype_policy

        # Qwen2 uses bias for Q/K/V and no bias for the output projection.
        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=True,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
        )
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            base=config.rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: Qwen2LayerCache | None = None,
    ) -> tuple[mx.array, Qwen2LayerCache]:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        k = self.k_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )
        v = self.v_proj(x).reshape(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )

        offset = 0 if cache is None else int(cache[0].shape[1])
        rotary_dtype = q.dtype if self.rotary_dtype_policy == "query" else mx.float32
        cos, sin = self.rotary_emb(offset, seq_len, dtype=rotary_dtype)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
        new_cache = (k, v)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        if mask is not None and mask.dtype != q.dtype:
            mask = mask.astype(q.dtype)
        out = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=self.scale,
            mask=mask,
        )
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(out), new_cache


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported Qwen2 activation {config.hidden_act!r}; expected 'silu'."
            )
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
        *,
        rotary_dtype_policy: Qwen2RotaryDtypePolicy = "float32",
    ):
        super().__init__()
        self.self_attn = Qwen2Attention(
            config,
            rotary_dtype_policy=rotary_dtype_policy,
        )
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: Qwen2LayerCache | None = None,
    ) -> tuple[mx.array, Qwen2LayerCache]:
        residual = x
        h, new_cache = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            cache=cache,
        )
        x = residual + h
        return x + self.mlp(self.post_attention_layernorm(x)), new_cache


@dataclass(frozen=True)
class Qwen2Output:
    last_hidden_state: mx.array
    cache: Qwen2KVCache

    @property
    def past_key_values(self) -> Qwen2KVCache:
        return self.cache


class Qwen2Model(nn.Module):
    """Qwen2 decoder backbone shared by speech model families."""

    def __init__(
        self,
        config: Qwen2Config,
        *,
        rotary_dtype_policy: Qwen2RotaryDtypePolicy = "float32",
    ):
        super().__init__()
        if config.model_type != "qwen2":
            raise ValueError(f"Unsupported Qwen2 model type: {config.model_type!r}.")
        if config.max_position_embeddings <= 0:
            raise ValueError("Qwen2 max_position_embeddings must be positive.")
        self.config = config
        self.rotary_dtype_policy = rotary_dtype_policy
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen2DecoderLayer(
                config,
                rotary_dtype_policy=rotary_dtype_policy,
            )
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def tied_logits(self, hidden_states: mx.array) -> mx.array:
        """Project hidden states with the token embedding weight."""

        return self.embed_tokens.as_linear(hidden_states)

    @staticmethod
    def _build_causal_mask(offset: int, seq_len: int) -> mx.array | None:
        if seq_len <= 1:
            return None
        key_len = offset + seq_len
        query_positions = mx.arange(offset, offset + seq_len, dtype=mx.int32)[:, None]
        key_positions = mx.arange(key_len, dtype=mx.int32)[None, :]
        allowed = query_positions >= key_positions
        mask = mx.where(
            allowed,
            mx.array(0.0, dtype=mx.float32),
            mx.array(float("-inf"), dtype=mx.float32),
        )
        return mask[None, None, :, :]

    def _prepare_inputs(
        self,
        *,
        input_ids: mx.array | None,
        inputs_embeds: mx.array | None,
    ) -> mx.array:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if input_ids is not None:
            if input_ids.ndim != 2:
                raise ValueError(
                    "Qwen2 input_ids must have shape (batch, sequence), "
                    f"got {input_ids.shape}."
                )
            return self.embed_tokens(input_ids)
        if inputs_embeds.ndim != 3:
            raise ValueError(
                "Qwen2 inputs_embeds must have shape (batch, sequence, hidden), "
                f"got {inputs_embeds.shape}."
            )
        if inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError(
                "Qwen2 input embedding width does not match hidden_size: "
                f"{inputs_embeds.shape[-1]} vs {self.config.hidden_size}."
            )
        return inputs_embeds

    def _cache_offset(
        self,
        cache: Qwen2KVCache | None,
        *,
        batch_size: int,
    ) -> int:
        if cache is None:
            return 0
        if len(cache) != len(self.layers):
            raise ValueError(
                "Qwen2 cache layer count does not match the model: "
                f"{len(cache)} vs {len(self.layers)}."
            )

        offset: int | None = None
        expected_tail = (self.config.num_key_value_heads, self.head_dim)
        for index, layer_cache in enumerate(cache):
            if not isinstance(layer_cache, tuple) or len(layer_cache) != 2:
                raise ValueError(f"Qwen2 cache layer {index} must be a (keys, values) tuple.")
            keys, values = layer_cache
            if keys.shape != values.shape or keys.ndim != 4:
                raise ValueError(
                    f"Qwen2 cache layer {index} has invalid key/value shapes: "
                    f"{keys.shape} vs {values.shape}."
                )
            if keys.shape[0] != batch_size or keys.shape[2:] != expected_tail:
                raise ValueError(
                    f"Qwen2 cache layer {index} has incompatible shape {keys.shape}."
                )
            layer_offset = int(keys.shape[1])
            if offset is None:
                offset = layer_offset
            elif layer_offset != offset:
                raise ValueError("Qwen2 cache layers must have the same sequence length.")
        return 0 if offset is None else offset

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: Qwen2KVCache | None = None,
    ) -> Qwen2Output:
        hidden_states = self._prepare_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len <= 0:
            raise ValueError("Qwen2 input sequence must not be empty.")
        offset = self._cache_offset(cache, batch_size=batch_size)
        if offset + seq_len > self.config.max_position_embeddings:
            raise ValueError(
                "Qwen2 sequence exceeds max_position_embeddings: "
                f"{offset + seq_len} > {self.config.max_position_embeddings}."
            )
        mask = self._build_causal_mask(offset, seq_len)

        new_cache: Qwen2KVCache = []
        for index, layer in enumerate(self.layers):
            layer_cache = None if cache is None else cache[index]
            hidden_states, next_cache = layer(
                hidden_states,
                mask=mask,
                cache=layer_cache,
            )
            new_cache.append(next_cache)

        return Qwen2Output(
            last_hidden_state=self.norm(hidden_states),
            cache=new_cache,
        )


__all__ = [
    "Qwen2Attention",
    "Qwen2Config",
    "Qwen2DecoderLayer",
    "Qwen2KVCache",
    "Qwen2LayerCache",
    "Qwen2MLP",
    "Qwen2Model",
    "Qwen2Output",
    "Qwen2RMSNorm",
    "Qwen2RotaryEmbedding",
    "Qwen2RotaryDtypePolicy",
]
