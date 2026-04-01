"""Step-Audio-EditX Step1 LM modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import mlx.core as mx
import mlx.nn as nn

from .config import Step1Config


def _linear_forward(
    linear: nn.Module,
    x: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
) -> mx.array:
    target_dtype = x.dtype if output_dtype is None else output_dtype
    if "Quantized" in type(linear).__name__:
        y = linear(x)
        return y if y.dtype == target_dtype else y.astype(target_dtype)

    weight = getattr(linear, "weight", None)
    if weight is None:
        y = linear(x)
        return y if y.dtype == target_dtype else y.astype(target_dtype)

    weight_cast = weight if weight.dtype == x.dtype else weight.astype(x.dtype)
    y = mx.matmul(x, weight_cast.T)
    bias = getattr(linear, "bias", None)
    if bias is not None:
        bias_cast = bias if bias.dtype == y.dtype else bias.astype(y.dtype)
        y = y + bias_cast
    return y.astype(target_dtype)


def _alibi_slopes(num_heads: int) -> mx.array:
    n = 2 ** math.floor(math.log2(num_heads))
    m0 = 2.0 ** (-8.0 / n)
    slopes = mx.power(
        mx.array(m0, dtype=mx.float32),
        mx.arange(1, n + 1, dtype=mx.float32),
    )
    if n < num_heads:
        m1 = 2.0 ** (-4.0 / n)
        extra = mx.power(
            mx.array(m1, dtype=mx.float32),
            mx.arange(1, 1 + 2 * (num_heads - n), 2, dtype=mx.float32),
        )
        slopes = mx.concatenate([slopes, extra], axis=0)
    return slopes


def build_sqrt_alibi_bias(
    query_len: int,
    key_len: int,
    num_heads: int,
    *,
    offset: int = 0,
    dtype: mx.Dtype = mx.float32,
) -> mx.array:
    """Build the shipped Step1 sqrt-ALiBi causal bias."""

    slopes = _alibi_slopes(num_heads)
    q_pos = mx.arange(offset, offset + query_len, dtype=mx.float32)[:, None]
    k_pos = mx.arange(key_len, dtype=mx.float32)[None, :]
    distance = q_pos - k_pos
    valid = distance >= 0
    distance = mx.where(valid, distance, mx.zeros_like(distance))
    bias = -mx.sqrt(distance)
    bias = bias[None, :, :] * slopes[:, None, None]
    neg_inf = mx.array(float("-inf"), dtype=mx.float32)
    bias = mx.where(valid[None, :, :], bias, neg_inf)
    return bias.astype(dtype)


def _repeat_kv_groups(x: mx.array, repeat: int) -> mx.array:
    """Repeat each KV group contiguously to match the trained GQA head ordering."""

    return mx.repeat(x, int(repeat), axis=2)


class Step1AttentionBias(Protocol):
    key: mx.array
    value: mx.array


@dataclass
class Step1LayerKVCache:
    """Mutable grouped KV cache stored in pre-repeat form."""

    keys: mx.array
    values: mx.array
    current_length: int = 0

    @staticmethod
    def allocate(
        *,
        batch_size: int,
        max_length: int,
        num_groups: int,
        head_dim: int,
        dtype: mx.Dtype,
    ) -> "Step1LayerKVCache":
        return Step1LayerKVCache(
            keys=mx.zeros((batch_size, max_length, num_groups, head_dim), dtype=dtype),
            values=mx.zeros((batch_size, max_length, num_groups, head_dim), dtype=dtype),
            current_length=0,
        )

    @property
    def max_length(self) -> int:
        return int(self.keys.shape[1])

    def append(self, new_k: mx.array, new_v: mx.array) -> None:
        if new_k.shape != new_v.shape:
            raise ValueError(
                f"Expected key/value shapes to match, got {new_k.shape} vs {new_v.shape}."
            )
        if new_k.ndim != 4:
            raise ValueError(
                "Expected key/value with shape (batch, seq, num_groups, head_dim), "
                f"got {new_k.shape}."
            )
        step = int(new_k.shape[1])
        end = self.current_length + step
        if end > self.max_length:
            raise ValueError(
                f"KV cache overflow: need {end} slots, cache only has {self.max_length}."
            )
        self.keys[:, self.current_length:end, :, :] = new_k
        self.values[:, self.current_length:end, :, :] = new_v
        self.current_length = end

    def get(self) -> tuple[mx.array, mx.array]:
        return (
            self.keys[:, : self.current_length, :, :],
            self.values[:, : self.current_length, :, :],
        )

    def reset(self) -> None:
        self.current_length = 0


@dataclass
class Step1KVCacheCollection:
    """Per-layer cache collection allocated once for Step1 decode."""

    layers: tuple[Step1LayerKVCache, ...]

    @property
    def current_length(self) -> int:
        return 0 if not self.layers else int(self.layers[0].current_length)

    @classmethod
    def allocate(
        cls,
        config: Step1Config,
        *,
        batch_size: int,
        max_length: int,
        dtype: mx.Dtype,
    ) -> "Step1KVCacheCollection":
        layers = tuple(
            Step1LayerKVCache.allocate(
                batch_size=batch_size,
                max_length=max_length,
                num_groups=config.num_attention_groups,
                head_dim=config.head_dim,
                dtype=dtype,
            )
            for _ in range(config.num_hidden_layers)
        )
        return cls(layers=layers)

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()


@dataclass(frozen=True)
class Step1LMOutput:
    last_hidden_state: mx.array
    cache: Step1KVCacheCollection | None


@dataclass(frozen=True)
class Step1CausalLMOutput:
    logits: mx.array
    hidden_states: mx.array
    cache: Step1KVCacheCollection | None


class Step1RMSNorm(nn.Module):
    """RMSNorm with float32 accumulation."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class Step1MLP(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = _linear_forward(self.gate_proj, x)
        up = _linear_forward(self.up_proj, x)
        hidden = nn.silu(gate) * up
        return _linear_forward(self.down_proj, hidden)


class Step1Attention(nn.Module):
    """Grouped-query attention with sqrt-ALiBi fallback math."""

    def __init__(self, config: Step1Config):
        super().__init__()
        if config.num_attention_heads % config.num_attention_groups != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_attention_groups "
                f"({config.num_attention_heads} vs {config.num_attention_groups})."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_groups = config.num_attention_groups
        self.head_dim = config.head_dim
        self.kv_repeat = config.kv_repeat
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_groups * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_groups * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        *,
        cache: Step1LayerKVCache | None = None,
    ) -> tuple[mx.array, Step1LayerKVCache | None]:
        batch_size, query_len, _ = x.shape

        q = _linear_forward(self.q_proj, x).reshape(
            batch_size, query_len, self.num_heads, self.head_dim,
        )
        k = _linear_forward(self.k_proj, x).reshape(
            batch_size, query_len, self.num_groups, self.head_dim,
        )
        v = _linear_forward(self.v_proj, x).reshape(
            batch_size, query_len, self.num_groups, self.head_dim,
        )

        cache_offset = 0
        if cache is not None:
            cache_offset = int(cache.current_length)
            cache.append(k, v)
            k, v = cache.get()

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        key_len = int(k.shape[2])
        bias = build_sqrt_alibi_bias(
            query_len,
            key_len,
            self.num_heads,
            offset=cache_offset,
            dtype=mx.float32,
        )

        q_grouped = q.reshape(batch_size, self.num_groups, self.kv_repeat, query_len, self.head_dim)
        k_grouped = k[:, :, None, :, :]
        scores = mx.matmul(
            q_grouped,
            k_grouped.transpose(0, 1, 2, 4, 3),
        ).reshape(batch_size, self.num_heads, query_len, key_len).astype(mx.float32)
        scores = (scores * self.scale) + bias[None, :, :, :]
        probs = mx.softmax(scores, axis=-1)
        probs_grouped = probs.astype(v.dtype).reshape(
            batch_size,
            self.num_groups,
            self.kv_repeat,
            query_len,
            key_len,
        )
        v_grouped = v[:, :, None, :, :]
        hidden = mx.matmul(probs_grouped, v_grouped).reshape(
            batch_size,
            self.num_heads,
            query_len,
            self.head_dim,
        )
        hidden = hidden.transpose(0, 2, 1, 3).reshape(batch_size, query_len, self.hidden_size)
        hidden = hidden.astype(x.dtype)
        return _linear_forward(self.o_proj, hidden), cache


class Step1Block(nn.Module):
    """Pre-norm decoder block."""

    def __init__(self, config: Step1Config):
        super().__init__()
        self.input_layernorm = Step1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Step1Attention(config)
        self.post_attention_layernorm = Step1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Step1MLP(config.hidden_size, config.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        *,
        cache: Step1LayerKVCache | None = None,
    ) -> tuple[mx.array, Step1LayerKVCache | None]:
        residual = x
        x = self.input_layernorm(x)
        h, updated_cache = self.self_attn(x, cache=cache)
        x = residual + h

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)
        return x, updated_cache


class Step1Model(nn.Module):
    """Decoder stack for the shipped Step1 checkpoint."""

    def __init__(self, config: Step1Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Step1Block(config) for _ in range(config.num_hidden_layers)]
        self.norm = Step1RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        cache: Step1KVCacheCollection | None = None,
    ) -> Step1LMOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        _ = attention_mask

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds
        for idx, layer in enumerate(self.layers):
            layer_cache = cache.layers[idx] if cache is not None and idx < len(cache.layers) else None
            hidden_states, _ = layer(
                hidden_states,
                cache=layer_cache,
            )

        hidden_states = self.norm(hidden_states)
        return Step1LMOutput(last_hidden_state=hidden_states, cache=cache)


class Step1ForCausalLM(nn.Module):
    """Causal LM wrapper matching the shipped checkpoint structure."""

    def __init__(self, config: Step1Config):
        super().__init__()
        self.config = config
        self.model = Step1Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def allocate_kv_cache(
        self,
        *,
        batch_size: int,
        max_length: int,
        dtype: mx.Dtype | None = None,
    ) -> Step1KVCacheCollection:
        resolved_dtype = self.get_input_embeddings().weight.dtype if dtype is None else dtype
        return Step1KVCacheCollection.allocate(
            self.config,
            batch_size=batch_size,
            max_length=max_length,
            dtype=resolved_dtype,
        )

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        attention_mask: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: Step1KVCacheCollection | None = None,
    ) -> Step1CausalLMOutput:
        outputs = self.model(
            input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache=cache,
        )
        logits = _linear_forward(self.lm_head, outputs.last_hidden_state)
        return Step1CausalLMOutput(
            logits=logits,
            hidden_states=outputs.last_hidden_state,
            cache=outputs.cache,
        )
