"""LTX multi-head attention with optional per-head gating.

Used by the embeddings connector (self-attention) and, in Stage 6, the audio
DiT (both self- and cross-attention). The connector flavor is self-only with
``apply_gated_attention=True``.

Reference: `.references/DramaBox/ltx2/ltx_core/model/transformer/attention.py:142-252`

Checkpoint key alignment: upstream `to_out` is `Sequential(Linear, Identity)`,
serialized as `to_out.0.{weight,bias}`. We mirror that by using a Python
list-of-modules — MLX serializes list-children as `.0.`, `.1.`, etc.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .rope import apply_split_rope


class _InnerRMSNorm(nn.Module):
    """RMSNorm with a learnable weight initialized to 1.0 (standard RMSNorm,
    unlike the Gemma 3 variant which uses ``(x_normed) * (1 + w)``)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight.astype(x.dtype), self.eps)


class LTXAttention(nn.Module):
    """Multi-head attention block matching the upstream LTX `Attention` class.

    Parameters:
        query_dim: hidden dim of the input.
        heads, dim_head: ``inner_dim = heads * dim_head``.
        context_dim: cross-attention context dim (default = ``query_dim``).
        apply_gated_attention: enables the per-head sigmoid gate.
        rope_type: ``"split"`` (used everywhere DramaBox).

    Saved keys (per block):
        - ``to_q.{weight,bias}`` / ``to_k.{weight,bias}`` / ``to_v.{weight,bias}``
          — `Linear(query_dim → inner_dim)` (etc.)
        - ``q_norm.weight`` / ``k_norm.weight`` — RMSNorm weights on full inner_dim
        - ``to_gate_logits.{weight,bias}`` — Linear(query_dim → heads), present iff gated
        - ``to_out.0.{weight,bias}`` — output Linear(inner_dim → query_dim)
    """

    def __init__(
        self,
        query_dim: int,
        heads: int,
        dim_head: int,
        *,
        context_dim: int | None = None,
        norm_eps: float = 1e-6,
        apply_gated_attention: bool = False,
        rope_type: str = "split",
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.rope_type = rope_type

        ctx = query_dim if context_dim is None else context_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(ctx, self.inner_dim, bias=True)
        self.to_v = nn.Linear(ctx, self.inner_dim, bias=True)

        self.q_norm = _InnerRMSNorm(self.inner_dim, eps=norm_eps)
        self.k_norm = _InnerRMSNorm(self.inner_dim, eps=norm_eps)

        # Optional per-head sigmoid gate
        if apply_gated_attention:
            self.to_gate_logits: nn.Linear | None = nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        # `to_out = Sequential(Linear, Identity)`. Use a list so MLX
        # serializes children as ``to_out.0.*`` (matching the checkpoint).
        self.to_out = [
            nn.Linear(self.inner_dim, query_dim, bias=True),
            nn.Identity(),
        ]

    def __call__(
        self,
        x: mx.array,
        *,
        context: mx.array | None = None,
        mask: mx.array | None = None,
        rope_cos_sin: tuple[mx.array, mx.array] | None = None,
        skip_self_attn: bool = False,
    ) -> mx.array:
        """Forward. ``mask`` is the *additive* attention bias matching MLX's
        ``mx.fast.scaled_dot_product_attention`` convention (broadcastable to
        ``(B, H, T_q, T_k)``).

        ``skip_self_attn`` is the STG (Spatio-Temporal Guidance) perturbation:
        the attention output becomes the raw value projection (each token attends
        only to itself, no QK softmax), while the per-head gate and ``to_out``
        still apply. Reference:
        ``.references/DramaBox/ltx2/ltx_core/model/transformer/attention.py:218-238``.
        """
        ctx = x if context is None else context

        if skip_self_attn:
            # STG passthrough: out = to_v(ctx); skip to_q/to_k/q_norm/k_norm/RoPE/SDPA.
            out = self.to_v(ctx)  # (B, T, inner_dim)
            B, T_q = out.shape[0], out.shape[1]
        else:
            q = self.to_q(x)  # (B, T, inner_dim)
            k = self.to_k(ctx)
            v = self.to_v(ctx)

            q = self.q_norm(q)
            k = self.k_norm(k)

            if rope_cos_sin is not None and self.rope_type == "split":
                cos, sin = rope_cos_sin
                q = apply_split_rope(q, cos, sin)
                k = apply_split_rope(k, cos, sin)

            B, T_q, _ = q.shape
            T_k = k.shape[1]
            q = q.reshape(B, T_q, self.heads, self.dim_head).transpose(0, 2, 1, 3)
            k = k.reshape(B, T_k, self.heads, self.dim_head).transpose(0, 2, 1, 3)
            v = v.reshape(B, T_k, self.heads, self.dim_head).transpose(0, 2, 1, 3)

            out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
            out = out.transpose(0, 2, 1, 3).reshape(B, T_q, self.heads * self.dim_head)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)  # (B, T, H)
            gates = 2.0 * mx.sigmoid(gate_logits)
            out = out.reshape(B, T_q, self.heads, self.dim_head)
            out = out * gates[..., None]
            out = out.reshape(B, T_q, self.heads * self.dim_head)

        # to_out.0 = Linear, to_out.1 = Identity (no-op)
        return self.to_out[0](out)


__all__ = ["LTXAttention"]
