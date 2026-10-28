"""Delay-conditioned Qwen2 decoder for Audio8."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .._cache import BoundedKVCache
from .._qwen2 import Qwen2Attention, Qwen2MLP, Qwen2RMSNorm, Qwen2RotaryEmbedding
from .config import Audio8TextConfig


Audio8TextCache = list[BoundedKVCache]


class Audio8TimeEmbedding(nn.Module):
    def __init__(self, hidden_size: int, theta: float = 10000.0) -> None:
        super().__init__()
        half = hidden_size // 2
        self.inv_freq = mx.exp(
            -math.log(theta) * mx.arange(half, dtype=mx.float32) / half
        )

    def __call__(self, num_delay_tokens: int, *, dtype: mx.Dtype) -> mx.array:
        if dtype not in (mx.bfloat16, mx.float16, mx.float32):
            raise ValueError(f"Audio8 time embedding requires a floating dtype, got {dtype}")
        delay = mx.array([num_delay_tokens], dtype=dtype)
        phase = delay[:, None] * self.inv_freq.astype(dtype)[None, :]
        return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)[None, :, :]


class Audio8AdaRMSNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, 32, bias=False)
        self.linear2 = nn.Linear(32, hidden_size, bias=False)

    def __call__(self, t_cond: mx.array) -> mx.array:
        return self.linear2(nn.gelu(self.linear1(t_cond)))


class Audio8DecoderLayer(nn.Module):
    def __init__(self, config: Audio8TextConfig) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(config, rotary_dtype_policy="query")
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.ada_rms_norm = Audio8AdaRMSNorm(config.hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        t_cond: mx.array,
        mask: mx.array | None,
        position_embeddings: tuple[mx.array, mx.array],
        cache: BoundedKVCache | None,
    ) -> tuple[mx.array, BoundedKVCache]:
        residual = hidden_states
        attended, new_cache = self.self_attn(
            self.input_layernorm(hidden_states),
            mask=mask,
            position_embeddings=position_embeddings,
            cache=cache,
        )
        hidden_states = residual + attended
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        modulation = self.ada_rms_norm(t_cond).astype(hidden_states.dtype)
        hidden_states = hidden_states * (1 + modulation)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, new_cache


class Audio8TextModel(nn.Module):
    def __init__(self, config: Audio8TextConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Audio8DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = Qwen2RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config.head_dim, config.rope_theta)

    def __call__(
        self,
        *,
        inputs_embeds: mx.array,
        t_cond: mx.array,
        cache: Audio8TextCache | None = None,
    ) -> tuple[mx.array, Audio8TextCache]:
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
            raise ValueError("Audio8 inputs_embeds must have shape [batch, sequence, hidden]")
        batch, sequence, _ = inputs_embeds.shape
        if sequence == 0:
            raise ValueError("Audio8 text input must not be empty")
        if t_cond.shape != (batch, 1, self.config.hidden_size):
            raise ValueError(
                f"Expected t_cond [{batch}, 1, {self.config.hidden_size}], got {t_cond.shape}"
            )
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("Audio8 text cache layer count mismatch")
        offset = 0 if cache is None else text_cache_length(cache)
        if offset + sequence > self.config.max_position_embeddings:
            raise ValueError("Audio8 text sequence exceeds max_position_embeddings")
        mask = _causal_mask(offset, sequence)
        position_embeddings = self.rotary_emb(
            offset, sequence, dtype=inputs_embeds.dtype
        )
        hidden_states = inputs_embeds
        new_cache: Audio8TextCache = []
        for index, layer in enumerate(self.layers):
            hidden_states, layer_cache = layer(
                hidden_states,
                t_cond=t_cond,
                mask=mask,
                position_embeddings=position_embeddings,
                cache=None if cache is None else cache[index],
            )
            new_cache.append(layer_cache)
        return self.norm(hidden_states), new_cache


@dataclass(frozen=True)
class Audio8TextOutput:
    logits: mx.array
    last_hidden_state: mx.array
    cache: Audio8TextCache


class Audio8LanguageModel(nn.Module):
    def __init__(self, config: Audio8TextConfig) -> None:
        super().__init__()
        self.model = Audio8TextModel(config)
        self.lm_head = (
            None
            if config.tie_word_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

    def __call__(
        self,
        *,
        inputs_embeds: mx.array,
        t_cond: mx.array,
        cache: Audio8TextCache | None = None,
        logits_to_keep: int = 1,
    ) -> Audio8TextOutput:
        hidden_states, new_cache = self.model(
            inputs_embeds=inputs_embeds, t_cond=t_cond, cache=cache
        )
        hidden = hidden_states[:, -logits_to_keep:, :] if logits_to_keep else hidden_states
        logits = (
            self.model.embed_tokens.as_linear(hidden)
            if self.lm_head is None
            else self.lm_head(hidden)
        )
        return Audio8TextOutput(
            logits=logits,
            last_hidden_state=hidden_states,
            cache=new_cache,
        )


def text_cache_length(cache: Audio8TextCache) -> int:
    lengths = {layer.offset for layer in cache}
    if len(lengths) != 1:
        raise ValueError("Audio8 text cache layers must have equal lengths")
    return lengths.pop() if lengths else 0


def _causal_mask(offset: int, sequence: int) -> mx.array | None:
    if sequence == 1:
        return None
    query_positions = mx.arange(offset, offset + sequence, dtype=mx.int32)[:, None]
    key_positions = mx.arange(offset + sequence, dtype=mx.int32)[None, :]
    return mx.where(
        query_positions >= key_positions,
        mx.array(0.0, dtype=mx.float32),
        mx.array(float("-inf"), dtype=mx.float32),
    )[None, None, :, :]
