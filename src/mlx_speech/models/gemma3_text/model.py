"""Pure-MLX Gemma 3 12B IT text-only forward pass.

Loads from the converted MLX 4-bit affine checkpoint and returns the full list
of hidden states (embedding output + 48 decoder layers). Operates under a
standard causal mask; DramaBox calls the upstream Gemma model with
``attention_mask`` (the 0/1 pad mask) on top of causal attention.

Gemma 3 quirks reflected here:

- RMSNorm is ``output = (x_fp32 * rsqrt(var + eps)) * (1 + w_fp32)``, cast back
  to the input dtype. Note the ``1 +`` — checkpoint norm weights are
  zero-centered.
- Embedding outputs are scaled by ``sqrt(hidden_size)``.
- Two RoPE families per layer: ``full_attention`` uses ``rope_theta`` with
  linear scaling (factor=8.0), ``sliding_attention`` uses
  ``rope_local_base_freq`` (10_000) with no scaling.
- For our use case ``seq_len ≤ sliding_window = 1024``, so the sliding-window
  mask collapses to ordinary causal. We use a single shared causal mask.
- Q and K each have their own per-head RMSNorm applied AFTER projection +
  reshape and BEFORE RoPE.
- ``query_pre_attn_scalar = 256`` → attention scale is ``1/sqrt(256)``.
- MLP is SwiGLU-style: ``down_proj(act(gate_proj) * up_proj)`` with
  ``gelu_pytorch_tanh`` (the tanh-approx GELU).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import GemmaTextConfig


# --------------------------------------------------------------------------- #
# RMSNorm — Gemma 3 variant
# --------------------------------------------------------------------------- #

def gemma_rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Gemma 3 RMSNorm: norm in fp32, multiply by ``1 + w`` in fp32, cast back.

    The ``+1`` accounts for the zero-centered checkpoint convention
    (``init.zeros_`` on the norm weight at init time).
    """
    orig_dtype = x.dtype
    x32 = x.astype(mx.float32)
    var = mx.mean(x32 * x32, axis=-1, keepdims=True)
    normed = x32 * mx.rsqrt(var + eps)
    out = normed * (1.0 + weight.astype(mx.float32))
    return out.astype(orig_dtype)


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return gemma_rms_norm(x, self.weight, self.eps)


# --------------------------------------------------------------------------- #
# Rotary Position Embedding
# --------------------------------------------------------------------------- #

def _inv_freq(head_dim: int, base: float, scaling_factor: float = 1.0) -> mx.array:
    """Standard RoPE inverse frequencies with optional linear scaling."""
    idx = mx.arange(0, head_dim, 2, dtype=mx.float32)
    inv = 1.0 / (base ** (idx / head_dim))
    if scaling_factor != 1.0:
        inv = inv / scaling_factor
    return inv


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _apply_rope(q: mx.array, k: mx.array, cos: mx.array, sin: mx.array) -> tuple[mx.array, mx.array]:
    """Apply RoPE to q, k. Shapes: q [B, Hq, L, D], k [B, Hkv, L, D],
    cos/sin [L, D]."""
    cos = cos[None, None, :, :]  # [1, 1, L, D]
    sin = sin[None, None, :, :]
    q_dtype, k_dtype = q.dtype, k.dtype
    q32, k32 = q.astype(mx.float32), k.astype(mx.float32)
    q_rot = (q32 * cos) + (_rotate_half(q32) * sin)
    k_rot = (k32 * cos) + (_rotate_half(k32) * sin)
    return q_rot.astype(q_dtype), k_rot.astype(k_dtype)


def _rope_cos_sin(
    seq_len: int,
    head_dim: int,
    base: float,
    scaling_factor: float = 1.0,
) -> tuple[mx.array, mx.array]:
    """Pre-compute cos/sin tables for positions ``[0, seq_len)``."""
    inv = _inv_freq(head_dim, base, scaling_factor)  # [D/2]
    pos = mx.arange(seq_len, dtype=mx.float32)  # [L]
    freqs = mx.outer(pos, inv)  # [L, D/2]
    emb = mx.concatenate([freqs, freqs], axis=-1)  # [L, D]
    return mx.cos(emb), mx.sin(emb)


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #

class Gemma3MLP(nn.Module):
    def __init__(self, config: GemmaTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        # gelu_pytorch_tanh is the tanh-approximation GELU
        self.act_fn = nn.GELU(approx="precise" if config.hidden_activation == "gelu" else "tanh")

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# Attention (GQA, Q/K norm, per-layer RoPE flavor)
# --------------------------------------------------------------------------- #

class Gemma3Attention(nn.Module):
    def __init__(self, config: GemmaTextConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scale = config.attention_scaling  # 1/sqrt(query_pre_attn_scalar)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Project, reshape to (B, H, L, D)
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Per-head RMSNorm on Q/K (over last dim D), then RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rope(q, k, cos, sin)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Decoder Layer
# --------------------------------------------------------------------------- #

class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config: GemmaTextConfig):
        super().__init__()
        self.self_attn = Gemma3Attention(config)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array | None,
    ) -> mx.array:
        # Attention sub-block
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, cos, sin, mask)
        h = self.post_attention_layernorm(h)
        x = residual + h

        # FFN sub-block
        residual = x
        h = self.pre_feedforward_layernorm(x)
        h = self.mlp(h)
        h = self.post_feedforward_layernorm(h)
        x = residual + h
        return x


# --------------------------------------------------------------------------- #
# Full Backbone
# --------------------------------------------------------------------------- #

@dataclass
class Gemma3Output:
    """Output of `Gemma3Model.__call__`.

    `hidden_states` is the list of 49 tensors: ``[embed_out, layer_0_out, …,
    layer_47_out]``. The final hidden state (after the trailing
    ``norm`` RMSNorm) is `last_hidden_state`. DramaBox consumes the full
    `hidden_states` list — not just the last one — so we always return both.
    """

    last_hidden_state: mx.array
    hidden_states: list[mx.array]


class Gemma3Model(nn.Module):
    """Gemma 3 text-only backbone, no LM head, no KV cache.

    Designed for a single forward pass per prompt: tokens come in left-padded
    to a fixed length (typically 1024), the model outputs all hidden states,
    and downstream consumers (e.g. DramaBox's `FeatureExtractorV2`) decide
    what to do with them.
    """

    def __init__(self, config: GemmaTextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [Gemma3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # `sqrt(hidden_size)` embed scaling factor (precomputed)
        self._embed_scale = float(math.sqrt(config.hidden_size))
        # Per-layer RoPE flavor selection.
        self._layer_types = config.layer_types()

    # ----- helpers ---------------------------------------------------------

    def _build_causal_mask(self, attention_mask: mx.array | None, seq_len: int, dtype: mx.Dtype) -> mx.array:
        """Causal mask + optional pad mask, returned as additive `[B,1,L,L]`.

        Padded positions get ``-finfo.max`` so the softmax zeros them out.
        We avoid `-inf` because some attention implementations propagate NaN
        when an entire row is `-inf`; this is the same trick the upstream
        `convert_to_additive_mask` uses.
        """
        # Causal `[L, L]` (lower-triangular True)
        q_idx = mx.arange(seq_len)[:, None]
        k_idx = mx.arange(seq_len)[None, :]
        causal = q_idx >= k_idx  # bool

        if attention_mask is not None:
            B = attention_mask.shape[0]
            keep = attention_mask.astype(mx.bool_)[:, None, :]  # [B, 1, L]
            full = causal[None, :, :] & keep  # [B, L, L]
        else:
            B = 1
            full = mx.broadcast_to(causal[None, :, :], (1, seq_len, seq_len))

        large_neg = mx.array(mx.finfo(dtype).min, dtype=dtype)
        zero = mx.array(0.0, dtype=dtype)
        additive = mx.where(full, zero, large_neg)
        return additive[:, None, :, :]  # [B, 1, L, L]

    # ----- forward ---------------------------------------------------------

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> Gemma3Output:
        """Forward pass returning all 49 hidden states.

        Args:
            input_ids: ``[B, L]`` int32/int64
            attention_mask: ``[B, L]`` 0/1, where 1 = valid token, 0 = pad.
                If None, no pad masking is applied (all tokens are valid).
        """
        B, L = input_ids.shape

        # Embed and scale
        x = self.embed_tokens(input_ids)
        x = x * mx.array(self._embed_scale, dtype=x.dtype)
        hidden_states: list[mx.array] = [x]

        # Pre-compute RoPE tables for both flavors (shared across layers of
        # the same type). With seq_len fixed at L this is cheap.
        cos_full, sin_full = _rope_cos_sin(
            seq_len=L,
            head_dim=self.config.head_dim,
            base=self.config.rope_theta,
            scaling_factor=self.config.rope_scaling.factor,
        )
        cos_sliding, sin_sliding = _rope_cos_sin(
            seq_len=L,
            head_dim=self.config.head_dim,
            base=self.config.rope_local_base_freq,
            scaling_factor=1.0,
        )

        mask = self._build_causal_mask(attention_mask, L, x.dtype)

        for layer, layer_type in zip(self.layers, self._layer_types):
            if layer_type == "full_attention":
                cos, sin = cos_full, sin_full
            else:
                cos, sin = cos_sliding, sin_sliding
            x = layer(x, cos, sin, mask)
            hidden_states.append(x)

        last = self.norm(x)
        return Gemma3Output(last_hidden_state=last, hidden_states=hidden_states)


__all__ = [
    "Gemma3Model",
    "Gemma3Output",
    "Gemma3DecoderLayer",
    "Gemma3Attention",
    "Gemma3MLP",
    "GemmaRMSNorm",
    "gemma_rms_norm",
]
