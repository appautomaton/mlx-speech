"""MLX model modules for MossTTSLocal."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .cache import GlobalKVCache, GlobalLayerKVCache, LocalKVCache, LocalLayerKVCache
from .config import MossTTSLocalConfig, Qwen3LanguageConfig

MOSS_TTS_ACTIVATION_DTYPE = mx.bfloat16


def _make_additive_attention_mask(
    seq_len: int,
    dtype: mx.Dtype,
    attention_mask: mx.array | None = None,
) -> mx.array:
    """Create an additive causal mask with optional key padding masking."""

    causal_mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len, dtype=dtype)
    if attention_mask is None:
        return causal_mask

    valid = attention_mask.astype(mx.bool_)
    if valid.ndim != 2:
        raise ValueError(
            f"Expected attention_mask with shape (batch, seq), got {attention_mask.shape}."
        )
    pad_mask = mx.where(
        valid[:, None, None, :],
        mx.array(0, dtype=dtype),
        mx.array(mx.finfo(dtype).min, dtype=dtype),
    )
    return causal_mask[None, None, :, :] + pad_mask


def _repeat_kv(x: mx.array, repeats: int) -> mx.array:
    """Repeat grouped key/value heads to match the query head count."""

    if repeats == 1:
        return x
    return mx.repeat(x, repeats, axis=1)


def _rotate_half(x: mx.array) -> mx.array:
    """Match Hugging Face Qwen3 rotary rotation order."""

    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


class MossTTSRotaryEmbedding(nn.Module):
    """Qwen3-compatible RoPE implementation."""

    def __init__(self, dim: int, *, base: float):
        super().__init__()
        self.dim = dim
        self.base = float(base)

    def cos_sin(
        self,
        x: mx.array,
        *,
        seq_len: int,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        inv_freq = 1.0 / (
            self.base
            ** (
                mx.arange(0, self.dim, 2, dtype=mx.float32)
                / mx.array(self.dim, dtype=mx.float32)
            )
        )
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)[None, :, None]
        freqs = positions * inv_freq[None, None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype)
        sin = mx.sin(emb).astype(x.dtype)
        return cos, sin

    def apply(
        self,
        query: mx.array,
        key: mx.array,
        *,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array]:
        seq_len = int(query.shape[-2])
        cos, sin = self.cos_sin(query, seq_len=seq_len, offset=offset)
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
        query = (query * cos) + (_rotate_half(query) * sin)
        key = (key * cos) + (_rotate_half(key) * sin)
        return query, key


def _linear_forward(
    linear: nn.Module,
    x: mx.array,
    *,
    output_dtype: mx.Dtype | None = None,
) -> mx.array:
    """Mixed-precision linear path.

    Quantized modules keep their own kernel path, but their outputs are cast
    back to the caller's activation dtype so hidden states stay in bf16.
    """

    target_dtype = x.dtype if output_dtype is None else output_dtype
    if "Quantized" in type(linear).__name__:
        y = linear(x)
        return y if y.dtype == target_dtype else y.astype(target_dtype)

    weight = getattr(linear, "weight", None)
    if weight is None:
        y = linear(x)
        return y if y.dtype == target_dtype else y.astype(target_dtype)

    y = mx.matmul(x.astype(mx.float32), weight.astype(mx.float32).T)
    bias = getattr(linear, "bias", None)
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y.astype(target_dtype)


class MossTTSRMSNorm(nn.Module):
    """Qwen3-style RMSNorm with float32 accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x_float = x.astype(mx.float32)
        variance = mx.mean(x_float * x_float, axis=-1, keepdims=True)
        x_norm = x_float * mx.rsqrt(variance + self.eps)
        return self.weight.astype(input_dtype) * x_norm.astype(input_dtype)


class MossTTSMLP(nn.Module):
    """SwiGLU MLP used by the global and local OpenMOSS blocks."""

    def __init__(self, input_size: int, ffn_hidden_size: int, output_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.up_proj = nn.Linear(input_size, ffn_hidden_size, bias=False)
        self.down_proj = nn.Linear(ffn_hidden_size, output_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gate = _linear_forward(self.gate_proj, x)
        up = _linear_forward(self.up_proj, x)
        hidden = nn.silu(gate) * up
        return _linear_forward(self.down_proj, hidden)


class MossTTSAttention(nn.Module):
    """Qwen3-style grouped-query attention with RoPE and q/k RMSNorm."""

    def __init__(self, config: Qwen3LanguageConfig, *, use_rope: bool = True):
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads: "
                f"{config.num_attention_heads} vs {config.num_key_value_heads}"
            )

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.effective_head_dim
        self.attention_output_size = self.num_attention_heads * self.head_dim
        self.kv_repeat = config.num_attention_heads // config.num_key_value_heads
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.attention_output_size,
            bias=False,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.attention_output_size,
            config.hidden_size,
            bias=False,
        )
        self.q_norm = MossTTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MossTTSRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = MossTTSRotaryEmbedding(self.head_dim, base=config.rope_theta)
        self.use_rope = use_rope
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _project_qkv(
        self,
        hidden_states: mx.array,
        *,
        offset: int = 0,
    ) -> tuple[mx.array, mx.array, mx.array]:
        batch_size, seq_len, _ = hidden_states.shape

        # Keep QKV projections and RoPE in float32 so cached decode stays numerically
        # aligned with the full-sequence reference path while hidden states remain bf16.
        query_states = _linear_forward(self.q_proj, hidden_states, output_dtype=mx.float32)
        key_states = _linear_forward(self.k_proj, hidden_states, output_dtype=mx.float32)
        value_states = _linear_forward(self.v_proj, hidden_states, output_dtype=mx.float32)

        query_states = query_states.reshape(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        key_states = key_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)
        value_states = value_states.reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(0, 2, 1, 3)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        if self.use_rope:
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
        use_causal_mask: bool = True,
    ) -> mx.array:
        batch_size, _, seq_len, _ = query_states.shape
        key_states = _repeat_kv(key_states, self.kv_repeat)
        value_states = _repeat_kv(value_states, self.kv_repeat)

        additive_mask = None
        if use_causal_mask:
            if int(key_states.shape[2]) != seq_len:
                raise ValueError(
                    "attention_mask is only supported when query and key lengths match in the "
                    f"full-sequence path, got q={seq_len} and k={int(key_states.shape[2])}."
                )
            additive_mask = _make_additive_attention_mask(
                seq_len,
                dtype=output_dtype,
                attention_mask=attention_mask,
            )

        attn_scores = (
            mx.matmul(
                query_states.astype(mx.float32),
                key_states.astype(mx.float32).transpose(0, 1, 3, 2),
            )
            * self.scale
        )
        if additive_mask is not None:
            attn_scores = attn_scores + additive_mask.astype(mx.float32)
        attn_weights = mx.softmax(attn_scores, axis=-1).astype(query_states.dtype)
        attn_output = mx.matmul(
            attn_weights.astype(mx.float32),
            value_states.astype(mx.float32),
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.attention_output_size
        )
        return _linear_forward(
            self.o_proj,
            attn_output,
            output_dtype=output_dtype,
        )

    def prefill(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: GlobalLayerKVCache | LocalLayerKVCache,
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
            use_causal_mask=True,
        )

    def decode_step(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: GlobalLayerKVCache | LocalLayerKVCache,
    ) -> mx.array:
        query_states, key_states, value_states = self._project_qkv(
            hidden_states,
            offset=layer_cache.current_length,
        )
        layer_cache.append(key_states, value_states)
        cached_keys, cached_values = layer_cache.get()
        return self._apply_attention(
            query_states,
            cached_keys,
            cached_values,
            output_dtype=hidden_states.dtype,
            attention_mask=None,
            use_causal_mask=False,
        )

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
            use_causal_mask=True,
        )


class Qwen3DecoderLayer(nn.Module):
    """One pre-norm Qwen3 decoder layer."""

    def __init__(self, config: Qwen3LanguageConfig, *, use_rope: bool = True):
        super().__init__()
        self.self_attn = MossTTSAttention(config, use_rope=use_rope)
        self.mlp = MossTTSMLP(
            input_size=config.hidden_size,
            ffn_hidden_size=config.intermediate_size,
            output_size=config.hidden_size,
        )
        self.input_layernorm = MossTTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MossTTSRMSNorm(
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
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states

    def prefill(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: GlobalLayerKVCache | LocalLayerKVCache,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn.prefill(
            self.input_layernorm(hidden_states),
            layer_cache=layer_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states

    def decode_step(
        self,
        hidden_states: mx.array,
        *,
        layer_cache: GlobalLayerKVCache | LocalLayerKVCache,
    ) -> mx.array:
        hidden_states = hidden_states + self.self_attn.decode_step(
            self.input_layernorm(hidden_states),
            layer_cache=layer_cache,
        )
        hidden_states = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states


@dataclass(frozen=True)
class Qwen3ModelOutput:
    """Return type for the Stage 2 global transformer."""

    last_hidden_state: mx.array
    hidden_states: tuple[mx.array, ...] | None = None


class Qwen3Model(nn.Module):
    """Minimal MLX implementation of the global Qwen3 backbone."""

    def __init__(self, config: Qwen3LanguageConfig, *, use_rope: bool = True):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )
        self.layers = [
            Qwen3DecoderLayer(config, use_rope=use_rope)
            for _ in range(config.num_hidden_layers)
        ]
        self.norm = MossTTSRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return Qwen3ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    def prefill(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        kv_cache: GlobalKVCache,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None

        for layer_index, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer.prefill(
                hidden_states,
                layer_cache=kv_cache.layers[layer_index],
                attention_mask=attention_mask,
            )

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return Qwen3ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    def decode_step(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        kv_cache: GlobalKVCache,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None

        for layer_index, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer.decode_step(
                hidden_states,
                layer_cache=kv_cache.layers[layer_index],
            )

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return Qwen3ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class MosiTTSModel(nn.Module):
    """Global multimodal embedding stack plus Qwen3 backbone."""

    def __init__(self, config: MossTTSLocalConfig):
        super().__init__()
        self.config = config
        self.text_pad_idx = config.pad_token_id
        self.speech_pad_idx = config.audio_pad_code
        self.channels = config.channels
        self.embedding_list = [
            nn.Embedding(config.vocab_size, config.hidden_size),
        ]
        for _ in range(1, self.channels):
            self.embedding_list.append(
                nn.Embedding(config.audio_embedding_vocab_size, config.hidden_size)
            )
        self.language_model = Qwen3Model(config.language_config)

    def _prepare_multi_modal_inputs(
        self,
        input_ids: mx.array,
        *,
        n_vq_for_inference: int | None = None,
    ) -> mx.array:
        batch_size, seq_length, channels = input_ids.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}.")

        if n_vq_for_inference is None:
            n_vq_for_inference = self.channels - 1
        n_vq_for_inference = max(1, min(self.channels - 1, int(n_vq_for_inference)))

        inputs_embeds = mx.zeros(
            (batch_size, seq_length, self.config.hidden_size),
            dtype=MOSS_TTS_ACTIVATION_DTYPE,
        )
        for layer_index in range(min(channels, 1 + n_vq_for_inference)):
            inputs_embeds = (
                inputs_embeds
                + self.embedding_list[layer_index](input_ids[..., layer_index])
            ).astype(MOSS_TTS_ACTIVATION_DTYPE)
        return inputs_embeds

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        n_vq_for_inference: int | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if input_ids is not None:
            inputs_embeds = self._prepare_multi_modal_inputs(
                input_ids,
                n_vq_for_inference=n_vq_for_inference,
            )
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    def prefill(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        n_vq_for_inference: int | None = None,
        max_cache_len: int,
        output_hidden_states: bool = False,
    ) -> tuple[Qwen3ModelOutput, GlobalKVCache]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if input_ids is not None:
            inputs_embeds = self._prepare_multi_modal_inputs(
                input_ids,
                n_vq_for_inference=n_vq_for_inference,
            )
        kv_cache = GlobalKVCache.allocate(
            self.language_model.config,
            batch_size=int(inputs_embeds.shape[0]),
            max_length=max_cache_len,
            dtype=mx.float32,
        )
        output = self.language_model.prefill(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
        )
        kv_cache.prompt_length = int(inputs_embeds.shape[1])
        return output, kv_cache

    def decode_step(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        kv_cache: GlobalKVCache,
        n_vq_for_inference: int | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if input_ids is not None:
            inputs_embeds = self._prepare_multi_modal_inputs(
                input_ids,
                n_vq_for_inference=n_vq_for_inference,
            )
        return self.language_model.decode_step(
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
        )


def _build_local_transformer_config(config: MossTTSLocalConfig) -> Qwen3LanguageConfig:
    payload = config.language_config.to_dict()
    payload["hidden_size"] = config.local_hidden_size
    payload["intermediate_size"] = config.local_ffn_hidden_size
    payload["num_hidden_layers"] = config.local_num_layers
    return Qwen3LanguageConfig.from_dict(payload)


class MossTTSLocalTransformer(nn.Module):
    """Local depth transformer used for RVQ token prediction."""

    def __init__(self, config: MossTTSLocalConfig):
        super().__init__()
        self.config = _build_local_transformer_config(config)
        self.layers = [
            Qwen3DecoderLayer(self.config, use_rope=False)
            for _ in range(self.config.num_hidden_layers)
        ]
        self.norm = MossTTSRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

    def __call__(
        self,
        *,
        inputs_embeds: mx.array,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        hidden_states = inputs_embeds
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return Qwen3ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )

    def decode_step(
        self,
        *,
        inputs_embeds: mx.array,
        kv_cache: LocalKVCache,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        hidden_states = inputs_embeds
        all_hidden_states: list[mx.array] | None = [] if output_hidden_states else None

        for layer_index, layer in enumerate(self.layers):
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)
            hidden_states = layer.decode_step(
                hidden_states,
                layer_cache=kv_cache.layers[layer_index],
            )

        hidden_states = self.norm(hidden_states)
        if all_hidden_states is not None:
            all_hidden_states.append(hidden_states)

        return Qwen3ModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=tuple(all_hidden_states) if all_hidden_states is not None else None,
        )


class MossTTSLocalModel(nn.Module):
    """Global + local modules needed for MossTTSLocal shape validation."""

    def __init__(self, config: MossTTSLocalConfig):
        super().__init__()
        self.config = config
        self.channels = config.channels
        self.model = MosiTTSModel(config)
        self.local_transformer_config = _build_local_transformer_config(config)
        self.local_transformer = MossTTSLocalTransformer(config)
        self.speech_embedding_to_local_mlp = MossTTSMLP(
            input_size=config.hidden_size,
            ffn_hidden_size=config.additional_mlp_ffn_hidden_size,
            output_size=config.local_hidden_size,
        )
        self.local_to_speech_embedding_mlps = [
            MossTTSMLP(
                input_size=config.local_hidden_size,
                ffn_hidden_size=config.additional_mlp_ffn_hidden_size,
                output_size=config.hidden_size,
            )
            for _ in range(self.channels)
        ]
        self.layer_norm_before_lm_heads = [
            MossTTSRMSNorm(config.hidden_size, eps=config.language_config.rms_norm_eps)
            for _ in range(self.channels)
        ]
        self.lm_heads = [nn.Linear(config.hidden_size, config.vocab_size, bias=False)]
        for _ in range(1, self.channels):
            self.lm_heads.append(
                nn.Linear(
                    config.hidden_size,
                    config.audio_embedding_vocab_size,
                    bias=False,
                )
            )

    def project_global_to_local(self, global_hidden_states: mx.array) -> mx.array:
        return self.speech_embedding_to_local_mlp(global_hidden_states)

    def forward_local_sequence(
        self,
        local_inputs_embeds: mx.array,
        *,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        return self.local_transformer(
            inputs_embeds=local_inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    def decode_local_step(
        self,
        local_input_embed: mx.array,
        *,
        kv_cache: LocalKVCache,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        return self.local_transformer.decode_step(
            inputs_embeds=local_input_embed,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
        )

    def project_local_outputs_to_logits(
        self,
        local_hidden_states: mx.array,
    ) -> tuple[mx.array, ...]:
        logits = []
        for proj, norm, head in zip(
            self.local_to_speech_embedding_mlps,
            self.layer_norm_before_lm_heads,
            self.lm_heads,
        ):
            logits.append(_linear_forward(head, norm(proj(local_hidden_states))))
        return tuple(logits)
