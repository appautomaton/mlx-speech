"""Qwen2 backbone for VibeVoice Large."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import Qwen2LanguageConfig

VIBEVOICE_ACTIVATION_DTYPE = mx.bfloat16


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #

class Qwen2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


# --------------------------------------------------------------------------- #
# Rotary Embeddings
# --------------------------------------------------------------------------- #

class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 1_000_000.0):
        super().__init__()
        self._dim = dim
        self._base = base

    def _inv_freq(self) -> mx.array:
        return 1.0 / (self._base ** (mx.arange(0, self._dim, 2, dtype=mx.float32) / self._dim))

    def __call__(self, offset: int, seq_len: int) -> tuple[mx.array, mx.array]:
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        inv_freq = self._inv_freq()
        freqs = mx.outer(positions, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array,
) -> tuple[mx.array, mx.array]:
    # q/k: (B, L, H, D), cos/sin: (L, D)
    cos = cos[None, :, None, :]  # (1, L, 1, D)
    sin = sin[None, :, None, :]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #

class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2LanguageConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Qwen2: q/k/v have bias, o does not
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(self.head_dim, base=config.rope_theta)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim)

        # RoPE with absolute positions
        offset = 0 if cache is None else cache[0].shape[1]
        cos, sin = self.rotary_emb(offset, L)
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # KV cache: concat-based (efficient in MLX's lazy evaluation model)
        if cache is not None:
            k_cache, v_cache = cache
            k = mx.concatenate([k_cache, k], axis=1)
            v = mx.concatenate([v_cache, v], axis=1)
        new_cache = (k, v)

        # Transpose for attention: (B, L, H, D) → (B, H, L, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)

        return self.o_proj(out), new_cache


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #

class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2LanguageConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Decoder Layer
# --------------------------------------------------------------------------- #

class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2LanguageConfig):
        super().__init__()
        self.self_attn = Qwen2Attention(config)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        residual = x
        x = self.input_layernorm(x)
        h, new_cache = self.self_attn(x, mask=mask, cache=cache)
        x = residual + h

        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        return x, new_cache


# --------------------------------------------------------------------------- #
# Full Qwen2 Model
# --------------------------------------------------------------------------- #

@dataclass
class Qwen2Output:
    last_hidden_state: mx.array
    cache: list[tuple[mx.array, mx.array]]


class Qwen2Model(nn.Module):
    """Qwen2 decoder backbone for VibeVoice Large."""

    def __init__(self, config: Qwen2LanguageConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _build_causal_mask(self, offset: int, seq_len: int) -> mx.array | None:
        if seq_len <= 1:
            return None
        k_len = offset + seq_len
        q_pos = mx.expand_dims(mx.arange(offset, offset + seq_len), axis=1)
        k_pos = mx.expand_dims(mx.arange(0, k_len), axis=0)
        allow = q_pos >= k_pos
        neg_inf = mx.array(float("-inf"), dtype=mx.float32)
        mask = mx.where(allow, mx.array(0.0, dtype=mx.float32), neg_inf)
        return mask[None, None, :, :]  # (1, 1, L, K)

    def __call__(
        self,
        *,
        inputs_embeds: mx.array,
        cache: list[tuple[mx.array, mx.array]] | None = None,
    ) -> Qwen2Output:
        h = inputs_embeds
        offset = 0 if cache is None or cache[0] is None else cache[0][0].shape[1]
        seq_len = h.shape[1]
        mask = self._build_causal_mask(offset, seq_len)

        new_caches: list[tuple[mx.array, mx.array]] = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            h, c = layer(h, mask=mask, cache=layer_cache)
            new_caches.append(c)

        h = self.norm(h)
        return Qwen2Output(last_hidden_state=h, cache=new_caches)
