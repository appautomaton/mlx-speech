"""Local Granite causal language model for Granite Speech ASR."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import GraniteSpeechTextConfig


def _repeat_kv(x: mx.array, repeats: int) -> mx.array:
    if repeats == 1:
        return x
    return mx.repeat(x, repeats, axis=1)


def _make_additive_causal_mask(
    query_len: int,
    key_len: int,
    *,
    offset: int,
    dtype: mx.Dtype,
) -> mx.array | None:
    if query_len == 1:
        return None
    query_positions = mx.arange(offset, offset + query_len)[:, None]
    key_positions = mx.arange(key_len)[None, :]
    allowed = query_positions >= key_positions
    return mx.where(
        allowed,
        mx.array(0, dtype=dtype),
        mx.array(mx.finfo(dtype).min, dtype=dtype),
    )[None, None, :, :]


@dataclass
class GraniteLayerKVCache:
    batch_size: int
    num_key_value_heads: int
    max_length: int
    head_dim: int
    dtype: mx.Dtype

    def __post_init__(self) -> None:
        self.keys = mx.zeros(
            (self.batch_size, self.num_key_value_heads, self.max_length, self.head_dim),
            dtype=self.dtype,
        )
        self.values = mx.zeros(
            (self.batch_size, self.num_key_value_heads, self.max_length, self.head_dim),
            dtype=self.dtype,
        )
        self.current_length = 0

    def append(self, keys: mx.array, values: mx.array) -> None:
        if keys.shape != values.shape:
            raise ValueError(f"Expected key/value shapes to match, got {keys.shape} vs {values.shape}")
        if keys.ndim != 4:
            raise ValueError(f"Expected key/value [B, H, T, D], got {keys.shape}")
        step = int(keys.shape[2])
        end = self.current_length + step
        if end > self.max_length:
            raise ValueError(f"KV cache overflow: need {end}, max_length is {self.max_length}")
        self.keys[:, :, self.current_length:end, :] = keys
        self.values[:, :, self.current_length:end, :] = values
        self.current_length = end

    def get(self) -> tuple[mx.array, mx.array]:
        return (
            self.keys[:, :, : self.current_length, :],
            self.values[:, :, : self.current_length, :],
        )


@dataclass
class GraniteKVCache:
    layers: tuple[GraniteLayerKVCache, ...]
    prompt_length: int = 0

    @property
    def current_length(self) -> int:
        return 0 if not self.layers else int(self.layers[0].current_length)

    @classmethod
    def allocate(
        cls,
        config: GraniteSpeechTextConfig,
        *,
        batch_size: int,
        max_length: int,
        dtype: mx.Dtype = mx.float32,
    ) -> "GraniteKVCache":
        head_dim = config.hidden_size // config.num_attention_heads
        layers = tuple(
            GraniteLayerKVCache(
                batch_size=batch_size,
                num_key_value_heads=config.num_key_value_heads,
                max_length=max_length,
                head_dim=head_dim,
                dtype=dtype,
            )
            for _ in range(config.num_hidden_layers)
        )
        return cls(layers=layers)


class GraniteAttention(nn.Module):
    def __init__(self, config: GraniteSpeechTextConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_repeat = config.num_attention_heads // config.num_key_value_heads
        self.scale = config.attention_multiplier
        bias = config.attention_bias

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=bias)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def _project_qkv(self, x: mx.array, *, offset: int) -> tuple[mx.array, mx.array, mx.array]:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_key_value_heads, self.head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)
        return q, k, v

    def __call__(
        self,
        x: mx.array,
        *,
        layer_cache: GraniteLayerKVCache | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        offset = 0 if layer_cache is None else layer_cache.current_length
        q, k, v = self._project_qkv(x, offset=offset)
        if layer_cache is not None:
            layer_cache.append(k, v)
            k, v = layer_cache.get()

        k = _repeat_kv(k, self.kv_repeat)
        v = _repeat_kv(v, self.kv_repeat)
        scores = mx.matmul(q.astype(mx.float32), k.astype(mx.float32).transpose(0, 1, 3, 2)) * self.scale
        mask = _make_additive_causal_mask(seq_len, int(k.shape[2]), offset=offset, dtype=mx.float32)
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1).astype(q.dtype)
        out = mx.matmul(weights.astype(mx.float32), v.astype(mx.float32))
        out = out.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)
        return self.o_proj(out).astype(x.dtype)


class GraniteMLP(nn.Module):
    def __init__(self, config: GraniteSpeechTextConfig):
        super().__init__()
        bias = config.mlp_bias
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class GraniteDecoderLayer(nn.Module):
    def __init__(self, config: GraniteSpeechTextConfig):
        super().__init__()
        self.self_attn = GraniteAttention(config)
        self.mlp = GraniteMLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.residual_multiplier = config.residual_multiplier

    def __call__(
        self,
        x: mx.array,
        *,
        layer_cache: GraniteLayerKVCache | None = None,
    ) -> mx.array:
        attn = self.self_attn(self.input_layernorm(x), layer_cache=layer_cache)
        h = x + attn * self.residual_multiplier
        mlp = self.mlp(self.post_attention_layernorm(h))
        return h + mlp * self.residual_multiplier


class GraniteBackbone(nn.Module):
    def __init__(self, config: GraniteSpeechTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [GraniteDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.embedding_multiplier = config.embedding_multiplier

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        kv_cache: GraniteKVCache | None = None,
    ) -> mx.array:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")

        if inputs_embeds is None:
            hidden_states = self.embed_tokens(input_ids)
        else:
            hidden_states = inputs_embeds
        hidden_states = hidden_states * self.embedding_multiplier

        for layer_index, layer in enumerate(self.layers):
            layer_cache = None if kv_cache is None else kv_cache.layers[layer_index]
            hidden_states = layer(hidden_states, layer_cache=layer_cache)
        return self.norm(hidden_states)


@dataclass(frozen=True)
class GraniteCausalLMOutput:
    logits: mx.array
    kv_cache: GraniteKVCache | None = None


class GraniteCausalLM(nn.Module):
    """Granite causal LM with local cache and input-embedding support."""

    def __init__(
        self,
        config: GraniteSpeechTextConfig,
        *,
        tie_word_embeddings: bool = False,
    ):
        super().__init__()
        self.config = config
        self.model = GraniteBackbone(config)
        self.tie_word_embeddings = tie_word_embeddings
        if not tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_scaling = config.logits_scaling

    @property
    def layers(self) -> list[GraniteDecoderLayer]:
        return self.model.layers

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        kv_cache: GraniteKVCache | None = None,
    ) -> mx.array:
        if input_embeddings is not None:
            if inputs_embeds is not None:
                raise ValueError("Specify only one of inputs_embeds or input_embeddings.")
            inputs_embeds = input_embeddings

        hidden = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )
        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(hidden)
        else:
            logits = self.lm_head(hidden)
        return logits / self.logits_scaling

    def prefill(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        max_cache_len: int,
    ) -> GraniteCausalLMOutput:
        batch_size = int((inputs_embeds if inputs_embeds is not None else input_ids).shape[0])
        cache = GraniteKVCache.allocate(
            self.config,
            batch_size=batch_size,
            max_length=max_cache_len,
            dtype=mx.float32,
        )
        logits = self(input_ids=input_ids, inputs_embeds=inputs_embeds, kv_cache=cache)
        cache.prompt_length = int(logits.shape[1])
        return GraniteCausalLMOutput(logits=logits, kv_cache=cache)

    def decode_step(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        kv_cache: GraniteKVCache,
    ) -> GraniteCausalLMOutput:
        logits = self(input_ids=input_ids, inputs_embeds=inputs_embeds, kv_cache=kv_cache)
        return GraniteCausalLMOutput(logits=logits, kv_cache=kv_cache)


def greedy_next_token(logits: mx.array) -> mx.array:
    """Return argmax token IDs from the final logits position."""
    return mx.argmax(logits[:, -1, :], axis=-1)
