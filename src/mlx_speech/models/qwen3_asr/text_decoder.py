"""Qwen3 text decoder pieces used by Qwen3-ASR."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen3ASRTextConfig


@dataclass
class Qwen3ASRTextLayerKVCache:
    batch_size: int
    num_kv_heads: int
    max_length: int
    head_dim: int
    dtype: mx.Dtype

    def __post_init__(self) -> None:
        self.keys = mx.zeros(
            (self.batch_size, self.num_kv_heads, self.max_length, self.head_dim),
            dtype=self.dtype,
        )
        self.values = mx.zeros(
            (self.batch_size, self.num_kv_heads, self.max_length, self.head_dim),
            dtype=self.dtype,
        )
        self.current_length = 0

    def append(self, key_states: mx.array, value_states: mx.array) -> None:
        if key_states.shape != value_states.shape:
            raise ValueError(
                f"Expected key/value shapes to match, got {key_states.shape} vs {value_states.shape}."
            )
        if key_states.ndim != 4:
            raise ValueError(
                "Expected key/value with shape (batch, kv_heads, seq, head_dim), "
                f"got {key_states.shape}."
            )
        step = int(key_states.shape[2])
        end = self.current_length + step
        if end > self.max_length:
            raise ValueError(f"Qwen3-ASR text KV cache overflow: need {end}, max {self.max_length}.")
        self.keys[:, :, self.current_length:end, :] = key_states
        self.values[:, :, self.current_length:end, :] = value_states
        self.current_length = end

    def get(self) -> tuple[mx.array, mx.array]:
        return (
            self.keys[:, :, : self.current_length, :],
            self.values[:, :, : self.current_length, :],
        )

    def reset(self) -> None:
        self.current_length = 0


@dataclass
class Qwen3ASRTextKVCache:
    layers: tuple[Qwen3ASRTextLayerKVCache, ...]
    prompt_length: int = 0

    @property
    def current_length(self) -> int:
        return 0 if not self.layers else int(self.layers[0].current_length)

    @classmethod
    def allocate(
        cls,
        config: Qwen3ASRTextConfig,
        *,
        batch_size: int,
        max_length: int,
        dtype: mx.Dtype = mx.float32,
    ) -> "Qwen3ASRTextKVCache":
        return cls(
            layers=tuple(
                Qwen3ASRTextLayerKVCache(
                    batch_size=batch_size,
                    num_kv_heads=config.num_key_value_heads,
                    max_length=max_length,
                    head_dim=config.head_dim,
                    dtype=dtype,
                )
                for _ in range(config.num_hidden_layers)
            )
        )

    def reset(self) -> None:
        self.prompt_length = 0
        for layer in self.layers:
            layer.reset()


@dataclass(frozen=True)
class Qwen3ASRTextModelOutput:
    last_hidden_state: mx.array
    past_key_values: Qwen3ASRTextKVCache | None = None
    hidden_states: tuple[mx.array, ...] | None = None


@dataclass(frozen=True)
class Qwen3ASRTextCausalLMOutput:
    logits: mx.array
    last_hidden_state: mx.array
    past_key_values: Qwen3ASRTextKVCache | None = None


class Qwen3ASRTextRMSNorm(nn.Module):
    """Qwen3 RMSNorm with float32 variance accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x_float = x.astype(mx.float32)
        variance = mx.mean(x_float * x_float, axis=-1, keepdims=True)
        normalized = x_float * mx.rsqrt(variance + self.eps)
        return self.weight.astype(input_dtype) * normalized.astype(input_dtype)


class Qwen3ASRTextRotaryEmbedding(nn.Module):
    """Qwen3 RoPE table using the ASR text config's rope_theta."""

    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.base = float(config.rope_theta)

    def cos_sin(
        self,
        *,
        seq_len: int,
        offset: int = 0,
        dtype: mx.Dtype = mx.float32,
    ) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / (
            self.base
            ** (
                mx.arange(0, self.head_dim, 2, dtype=mx.float32)
                / mx.array(self.head_dim, dtype=mx.float32)
            )
        )
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)[:, None]
        freqs = positions * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)

    def apply(
        self,
        query: mx.array,
        key: mx.array,
        *,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        seq_len = int(query.shape[-2])
        cos, sin = self.cos_sin(seq_len=seq_len, offset=offset, dtype=query.dtype)
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
        return (
            (query * cos) + (_rotate_half(query) * sin),
            (key * cos) + (_rotate_half(key) * sin),
        )


class Qwen3ASRTextMLP(nn.Module):
    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.hidden_act = config.hidden_act

    def __call__(self, x: mx.array) -> mx.array:
        gate = _linear_forward(self.gate_proj, x)
        up = _linear_forward(self.up_proj, x)
        hidden = _activation(gate, self.hidden_act) * up
        return _linear_forward(self.down_proj, hidden)


class Qwen3ASRTextAttention(nn.Module):
    """Qwen3 grouped-query attention with Q/K RMSNorm and RoPE."""

    def __init__(self, config: Qwen3ASRTextConfig, *, layer_idx: int):
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{config.num_attention_heads} vs {config.num_key_value_heads}."
            )
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_output_size = self.num_attention_heads * self.head_dim
        self.kv_repeat = self.num_attention_heads // self.num_key_value_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.attention_output_size,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.attention_output_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3ASRTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3ASRTextRotaryEmbedding(config)

    def _project_qkv(
        self,
        hidden_states: mx.array,
        *,
        offset: int,
    ) -> tuple[mx.array, mx.array, mx.array]:
        batch_size, seq_len, _ = hidden_states.shape
        query_states = _linear_forward(self.q_proj, hidden_states, output_dtype=mx.float32)
        key_states = _linear_forward(self.k_proj, hidden_states, output_dtype=mx.float32)
        value_states = _linear_forward(self.v_proj, hidden_states, output_dtype=mx.float32)

        query_states = query_states.reshape(
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = self.rotary_emb.apply(
            query_states,
            key_states,
            offset=offset,
        )
        return query_states, key_states, value_states

    def _apply_attention(
        self,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        *,
        output_dtype: mx.Dtype,
        attention_mask: mx.array | None = None,
        query_offset: int = 0,
        use_causal_mask: bool = True,
    ) -> mx.array:
        batch_size, _, query_len, _ = query_states.shape
        key_states = _repeat_kv(key_states, self.kv_repeat)
        value_states = _repeat_kv(value_states, self.kv_repeat)
        key_len = int(key_states.shape[2])

        scores = (
            mx.matmul(
                query_states.astype(mx.float32),
                key_states.astype(mx.float32).transpose(0, 1, 3, 2),
            )
            * self.scale
        )
        additive_mask = _make_additive_attention_mask(
            query_len=query_len,
            key_len=key_len,
            query_offset=query_offset,
            dtype=mx.float32,
            attention_mask=attention_mask,
            use_causal_mask=use_causal_mask,
        )
        if additive_mask is not None:
            scores = scores + additive_mask
        weights = mx.softmax(scores, axis=-1).astype(query_states.dtype)
        attn_output = mx.matmul(weights.astype(mx.float32), value_states.astype(mx.float32))
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            query_len,
            self.attention_output_size,
        )
        return _linear_forward(self.o_proj, attn_output, output_dtype=output_dtype)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None = None,
        offset: int = 0,
    ) -> mx.array:
        query_states, key_states, value_states = self._project_qkv(
            hidden_states,
            offset=offset,
        )
        return self._apply_attention(
            query_states,
            key_states,
            value_states,
            output_dtype=hidden_states.dtype,
            attention_mask=attention_mask,
            query_offset=offset,
            use_causal_mask=True,
        )

    def prefill(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: Qwen3ASRTextLayerKVCache,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        query_states, key_states, value_states = self._project_qkv(hidden_states, offset=0)
        layer_cache.append(key_states, value_states)
        cached_keys, cached_values = layer_cache.get()
        return self._apply_attention(
            query_states,
            cached_keys,
            cached_values,
            output_dtype=hidden_states.dtype,
            attention_mask=attention_mask,
            query_offset=0,
            use_causal_mask=True,
        )

    def decode_step(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: Qwen3ASRTextLayerKVCache,
    ) -> mx.array:
        offset = layer_cache.current_length
        query_states, key_states, value_states = self._project_qkv(
            hidden_states,
            offset=offset,
        )
        layer_cache.append(key_states, value_states)
        cached_keys, cached_values = layer_cache.get()
        return self._apply_attention(
            query_states,
            cached_keys,
            cached_values,
            output_dtype=hidden_states.dtype,
            attention_mask=None,
            query_offset=offset,
            use_causal_mask=True,
        )


class Qwen3ASRTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3ASRTextConfig, *, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3ASRTextAttention(config, layer_idx=layer_idx)
        self.mlp = Qwen3ASRTextMLP(config)
        self.input_layernorm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3ASRTextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None = None,
        offset: int = 0,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            attention_mask=attention_mask,
            offset=offset,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    def prefill(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: Qwen3ASRTextLayerKVCache,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn.prefill(
            self.input_layernorm(hidden_states),
            layer_cache=layer_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states

    def decode_step(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: Qwen3ASRTextLayerKVCache,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn.decode_step(
            self.input_layernorm(hidden_states),
            layer_cache=layer_cache,
        )
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states


class Qwen3ASRTextModel(nn.Module):
    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3ASRTextDecoderLayer(config, layer_idx=layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3ASRTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ASRTextModelOutput:
        hidden_states = self._prepare_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None
        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        return Qwen3ASRTextModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    def prefill(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        kv_cache: Qwen3ASRTextKVCache,
        output_hidden_states: bool = False,
    ) -> Qwen3ASRTextModelOutput:
        hidden_states = self._prepare_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer.prefill(
                hidden_states,
                layer_cache=kv_cache.layers[layer_idx],
                attention_mask=attention_mask,
            )
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        kv_cache.prompt_length = int(hidden_states.shape[1])
        return Qwen3ASRTextModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=kv_cache,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    def decode_step(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        kv_cache: Qwen3ASRTextKVCache,
        output_hidden_states: bool = False,
    ) -> Qwen3ASRTextModelOutput:
        hidden_states = self._prepare_inputs(input_ids=input_ids, inputs_embeds=inputs_embeds)
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer.decode_step(
                hidden_states,
                layer_cache=kv_cache.layers[layer_idx],
            )
        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)
        return Qwen3ASRTextModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=kv_cache,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    def _prepare_inputs(
        self,
        *,
        input_ids: mx.array | None,
        inputs_embeds: mx.array | None,
    ) -> mx.array:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if inputs_embeds is not None:
            return inputs_embeds
        return self.embed_tokens(input_ids)


class Qwen3ASRTextForCausalLM(nn.Module):
    def __init__(self, config: Qwen3ASRTextConfig):
        super().__init__()
        self.config = config
        self.model = Qwen3ASRTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.tie_word_embeddings = bool(config.extra.get("tie_word_embeddings", False))
        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
    ) -> Qwen3ASRTextCausalLMOutput:
        model_output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        logits = _linear_forward(self.lm_head, model_output.last_hidden_state, output_dtype=mx.float32)
        return Qwen3ASRTextCausalLMOutput(
            logits=logits,
            last_hidden_state=model_output.last_hidden_state,
            past_key_values=model_output.past_key_values,
        )

    def prefill(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        kv_cache: Qwen3ASRTextKVCache,
    ) -> Qwen3ASRTextCausalLMOutput:
        model_output = self.model.prefill(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        logits = _linear_forward(self.lm_head, model_output.last_hidden_state, output_dtype=mx.float32)
        return Qwen3ASRTextCausalLMOutput(
            logits=logits,
            last_hidden_state=model_output.last_hidden_state,
            past_key_values=kv_cache,
        )

    def decode_step(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        kv_cache: Qwen3ASRTextKVCache,
    ) -> Qwen3ASRTextCausalLMOutput:
        model_output = self.model.decode_step(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        logits = _linear_forward(self.lm_head, model_output.last_hidden_state, output_dtype=mx.float32)
        return Qwen3ASRTextCausalLMOutput(
            logits=logits,
            last_hidden_state=model_output.last_hidden_state,
            past_key_values=kv_cache,
        )


def _activation(x: mx.array, name: str) -> mx.array:
    if name == "silu":
        return nn.silu(x)
    if name == "gelu":
        return nn.gelu(x)
    raise ValueError(f"Unsupported Qwen3-ASR text activation: {name}")


def _linear_forward(
    linear: nn.Module,
    x: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
) -> mx.array:
    target_dtype = x.dtype if output_dtype is None else output_dtype
    weight = getattr(linear, "weight", None)
    if weight is None:
        y = linear(x)
        return y if y.dtype == target_dtype else y.astype(target_dtype)
    y = mx.matmul(x.astype(mx.float32), weight.astype(mx.float32).T)
    bias = getattr(linear, "bias", None)
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y.astype(target_dtype)


def _repeat_kv(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    return mx.repeat(x, repeats, axis=1)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _make_additive_attention_mask(
    *,
    query_len: int,
    key_len: int,
    query_offset: int,
    dtype: mx.Dtype,
    attention_mask: mx.array | None = None,
    use_causal_mask: bool,
) -> mx.array | None:
    mask: mx.array | None = None
    if use_causal_mask:
        query_positions = mx.arange(query_offset, query_offset + query_len, dtype=mx.int32)
        key_positions = mx.arange(key_len, dtype=mx.int32)
        future = key_positions[None, :] > query_positions[:, None]
        mask = mx.where(
            future,
            mx.array(mx.finfo(dtype).min, dtype=dtype),
            mx.array(0, dtype=dtype),
        )[None, None, :, :]

    if attention_mask is not None:
        valid = attention_mask.astype(mx.bool_)
        if valid.ndim != 2:
            raise ValueError(f"Expected attention_mask [batch, seq], got {attention_mask.shape}.")
        if int(valid.shape[1]) != key_len:
            raise ValueError(
                f"attention_mask sequence length {valid.shape[1]} does not match key length {key_len}."
            )
        pad_mask = mx.where(
            valid[:, None, None, :],
            mx.array(0, dtype=dtype),
            mx.array(mx.finfo(dtype).min, dtype=dtype),
        )
        mask = pad_mask if mask is None else mask + pad_mask
    return mask
