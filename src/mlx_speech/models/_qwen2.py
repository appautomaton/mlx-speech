"""Family-neutral pure-MLX Qwen2 decoder trunk."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Protocol

import mlx.core as mx
import mlx.nn as nn

from ._cache import BoundedKVCache


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


Qwen2LayerCache = BoundedKVCache
Qwen2KVCache = list[Qwen2LayerCache]
Qwen2RotaryDtypePolicy = Literal["float32", "query"]
_LegacyQwen2LayerCache = tuple[mx.array, mx.array]
_QWEN2_CACHE_GROWTH = 256


class _FusedQwenProjection(nn.Module):
    """Inference-only, weight-exact concatenation of Qwen projections."""

    def __init__(self, layers: tuple[nn.Module, ...]):
        super().__init__()
        if not layers:
            raise ValueError("fused Qwen projection requires source layers")
        quantized = isinstance(layers[0], nn.QuantizedLinear)
        expected_type = nn.QuantizedLinear if quantized else nn.Linear
        if any(not isinstance(layer, expected_type) for layer in layers):
            raise TypeError("fused Qwen projections must use one Linear type")
        has_bias = "bias" in layers[0]
        input_width = int(layers[0].weight.shape[1])
        if any(
            int(layer.weight.shape[1]) != input_width
            or layer.weight.dtype != layers[0].weight.dtype
            or ("bias" in layer) != has_bias
            for layer in layers
        ):
            raise ValueError("fused Qwen source projections are incompatible")

        object.__setattr__(
            self,
            "output_widths",
            tuple(int(layer.weight.shape[0]) for layer in layers),
        )
        self.weight = mx.concatenate(tuple(layer.weight for layer in layers), axis=0)
        if has_bias:
            self.bias = mx.concatenate(tuple(layer.bias for layer in layers), axis=0)
        self.quantized = quantized
        if quantized:
            first = layers[0]
            self.group_size = int(first.group_size)
            self.bits = int(first.bits)
            self.mode = str(first.mode)
            if any(
                int(layer.group_size) != self.group_size
                or int(layer.bits) != self.bits
                or str(layer.mode) != self.mode
                for layer in layers[1:]
            ):
                raise ValueError("fused Qwen quantization parameters differ")
            self.scales = mx.concatenate(
                tuple(layer.scales for layer in layers), axis=0
            )
            quant_biases = tuple(layer.get("biases") for layer in layers)
            if any(biases is None for biases in quant_biases):
                if not all(biases is None for biases in quant_biases):
                    raise ValueError("fused Qwen quantization biases differ")
            else:
                self.biases = mx.concatenate(quant_biases, axis=0)
        self.freeze()

    def __call__(self, value: mx.array) -> mx.array:
        if self.quantized:
            value = mx.quantized_matmul(
                value,
                self["weight"],
                scales=self["scales"],
                biases=self.get("biases"),
                transpose=True,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
            )
            if "bias" in self:
                value = value + self["bias"]
            return value
        if "bias" in self:
            return mx.addmm(self["bias"], value, self["weight"].T)
        return value @ self["weight"].T

    def split(self, value: mx.array) -> tuple[mx.array, ...]:
        projected = self(value)
        boundaries = []
        offset = 0
        for width in self.output_widths[:-1]:
            offset += width
            boundaries.append(offset)
        return tuple(mx.split(projected, boundaries, axis=-1))


@lru_cache(maxsize=16)
def _qwen2_inv_freq(dim: int, base: float) -> mx.array:
    exponent = mx.arange(0, dim, 2, dtype=mx.float32) / dim
    frequencies = 1.0 / (base**exponent)
    mx.eval(frequencies)
    return frequencies


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
        _qwen2_inv_freq(self._dim, self._base)

    def _inv_freq(self) -> mx.array:
        return _qwen2_inv_freq(self._dim, self._base)

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
            raise ValueError(
                f"Qwen2 RoPE sequence length must be positive, got {seq_len}."
            )
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
        self.max_position_embeddings = config.max_position_embeddings

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

    def fuse_qkv_for_inference(self) -> None:
        """Replace three decode projections with one weight-exact MLX matmul."""

        if getattr(self, "qkv_proj", None) is not None:
            return
        fused = _FusedQwenProjection((self.q_proj, self.k_proj, self.v_proj))
        mx.eval(fused.parameters())
        self.qkv_proj = fused
        del self.q_proj
        del self.k_proj
        del self.v_proj

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        cache: Qwen2LayerCache | _LegacyQwen2LayerCache | None = None,
        cache_capacity: int | None = None,
        max_cache_capacity: int | None = None,
        cache_growth_step: int | None = None,
    ) -> tuple[mx.array, Qwen2LayerCache]:
        batch_size, seq_len, _ = x.shape

        qkv_proj = getattr(self, "qkv_proj", None)
        if qkv_proj is None:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        else:
            q, k, v = qkv_proj.split(x)

        q = q.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        )
        k = k.reshape(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )
        v = v.reshape(
            batch_size,
            seq_len,
            self.num_kv_heads,
            self.head_dim,
        )

        if cache is None:
            offset = 0
        elif isinstance(cache, BoundedKVCache):
            offset = cache.offset
        else:
            offset = int(cache[0].shape[1])
        if offset + seq_len > self.max_position_embeddings:
            raise ValueError(
                "Qwen2 attention cache exceeds max_position_embeddings: "
                f"{offset + seq_len} > {self.max_position_embeddings}."
            )
        if position_embeddings is None:
            rotary_dtype = (
                q.dtype if self.rotary_dtype_policy == "query" else mx.float32
            )
            cos, sin = self.rotary_emb(offset, seq_len, dtype=rotary_dtype)
        else:
            cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        if isinstance(cache, BoundedKVCache):
            cache.append(k, v)
            new_cache = cache
        else:
            if cache_capacity is None:
                cache_capacity = min(
                    (
                        (offset + seq_len + _QWEN2_CACHE_GROWTH - 1)
                        // _QWEN2_CACHE_GROWTH
                    )
                    * _QWEN2_CACHE_GROWTH,
                    self.max_position_embeddings,
                )
                max_cache_capacity = self.max_position_embeddings
                cache_growth_step = _QWEN2_CACHE_GROWTH
            new_cache = BoundedKVCache.allocate(
                batch_size=batch_size,
                capacity=cache_capacity,
                num_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                key_dtype=k.dtype,
                value_dtype=v.dtype,
                max_capacity=max_cache_capacity,
                growth_step=cache_growth_step,
            )
            if cache is not None:
                new_cache.append(*cache)
            new_cache.append(k, v)
        k, v = new_cache.fetch()

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

    def fuse_gate_up_for_inference(self) -> None:
        """Replace the two SwiGLU input projections with one MLX matmul."""

        if getattr(self, "gate_up_proj", None) is not None:
            return
        fused = _FusedQwenProjection((self.gate_proj, self.up_proj))
        mx.eval(fused.parameters())
        self.gate_up_proj = fused
        del self.gate_proj
        del self.up_proj

    def __call__(self, x: mx.array) -> mx.array:
        gate_up_proj = getattr(self, "gate_up_proj", None)
        if gate_up_proj is None:
            gate, up = self.gate_proj(x), self.up_proj(x)
        else:
            gate, up = gate_up_proj.split(x)
        return self.down_proj(nn.silu(gate) * up)


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
        position_embeddings: tuple[mx.array, mx.array] | None = None,
        cache: Qwen2LayerCache | _LegacyQwen2LayerCache | None = None,
        cache_capacity: int | None = None,
        max_cache_capacity: int | None = None,
        cache_growth_step: int | None = None,
    ) -> tuple[mx.array, Qwen2LayerCache]:
        residual = x
        h, new_cache = self.self_attn(
            self.input_layernorm(x),
            mask=mask,
            position_embeddings=position_embeddings,
            cache=cache,
            cache_capacity=cache_capacity,
            max_cache_capacity=max_cache_capacity,
            cache_growth_step=cache_growth_step,
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
        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            base=config.rope_theta,
        )
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

    def fuse_for_inference(self) -> None:
        """Install the low-dispatch projection layout after checkpoint loading."""

        for layer in self.layers:
            layer.self_attn.fuse_qkv_for_inference()
            layer.mlp.fuse_gate_up_for_inference()

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
        cache: Qwen2KVCache | list[_LegacyQwen2LayerCache] | None,
        *,
        batch_size: int,
        exact_capacity: int | None,
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
            if isinstance(layer_cache, BoundedKVCache):
                keys, values = layer_cache.fetch()
                if (
                    layer_cache.capacity > self.config.max_position_embeddings
                    or int(layer_cache.max_capacity)
                    > self.config.max_position_embeddings
                ):
                    raise ValueError(
                        f"Qwen2 cache layer {index} exceeds max_position_embeddings."
                    )
                if exact_capacity is not None and (
                    layer_cache.capacity != exact_capacity
                    or int(layer_cache.max_capacity) != exact_capacity
                ):
                    raise ValueError(
                        "Qwen2 cache capacity differs from the requested exact "
                        f"capacity: {layer_cache.capacity} vs {exact_capacity}."
                    )
            elif isinstance(layer_cache, tuple) and len(layer_cache) == 2:
                keys, values = layer_cache
            else:
                raise ValueError(
                    f"Qwen2 cache layer {index} must contain keys and values."
                )
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
                raise ValueError(
                    "Qwen2 cache layers must have the same sequence length."
                )
        return 0 if offset is None else offset

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: Qwen2KVCache | list[_LegacyQwen2LayerCache] | None = None,
        cache_capacity: int | None = None,
    ) -> Qwen2Output:
        hidden_states = self._prepare_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
        )
        batch_size, seq_len, _ = hidden_states.shape
        if seq_len <= 0:
            raise ValueError("Qwen2 input sequence must not be empty.")
        if cache_capacity is not None:
            if cache_capacity <= 0:
                raise ValueError("Qwen2 cache_capacity must be positive.")
            if cache_capacity > self.config.max_position_embeddings:
                raise ValueError(
                    "Qwen2 cache_capacity exceeds max_position_embeddings: "
                    f"{cache_capacity} > {self.config.max_position_embeddings}."
                )
        offset = self._cache_offset(
            cache,
            batch_size=batch_size,
            exact_capacity=cache_capacity,
        )
        if offset + seq_len > self.config.max_position_embeddings:
            raise ValueError(
                "Qwen2 sequence exceeds max_position_embeddings: "
                f"{offset + seq_len} > {self.config.max_position_embeddings}."
            )
        if cache_capacity is not None and offset + seq_len > cache_capacity:
            raise ValueError(
                "Qwen2 sequence exceeds cache_capacity: "
                f"{offset + seq_len} > {cache_capacity}."
            )
        if cache is not None:
            for layer_cache in cache:
                if isinstance(layer_cache, BoundedKVCache):
                    layer_cache.validate_append_length(seq_len)
        mask = self._build_causal_mask(offset, seq_len)
        rotary_dtype = (
            hidden_states.dtype
            if self.rotary_dtype_policy == "query"
            else mx.float32
        )
        position_embeddings = self.rotary_emb(
            offset,
            seq_len,
            dtype=rotary_dtype,
        )

        if cache_capacity is None:
            initial_capacity = min(
                ((offset + seq_len + _QWEN2_CACHE_GROWTH - 1) // _QWEN2_CACHE_GROWTH)
                * _QWEN2_CACHE_GROWTH,
                self.config.max_position_embeddings,
            )
            max_cache_capacity = self.config.max_position_embeddings
            cache_growth_step = _QWEN2_CACHE_GROWTH
        else:
            initial_capacity = cache_capacity
            max_cache_capacity = cache_capacity
            cache_growth_step = None

        mutable_offsets = (
            tuple(
                (layer_cache, layer_cache.offset)
                for layer_cache in cache
                if isinstance(layer_cache, BoundedKVCache)
            )
            if cache is not None
            else ()
        )
        new_cache: Qwen2KVCache = []
        try:
            for index, layer in enumerate(self.layers):
                layer_cache = None if cache is None else cache[index]
                hidden_states, next_cache = layer(
                    hidden_states,
                    mask=mask,
                    position_embeddings=position_embeddings,
                    cache=layer_cache,
                    cache_capacity=initial_capacity,
                    max_cache_capacity=max_cache_capacity,
                    cache_growth_step=cache_growth_step,
                )
                new_cache.append(next_cache)
        except Exception:
            for layer_cache, prior_offset in mutable_offsets:
                layer_cache.restore_offset(prior_offset)
            raise

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
