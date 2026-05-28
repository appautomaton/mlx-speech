"""LTX DiT block — one of 48 transformer blocks.

For DramaBox (audio-only, ``cross_attention_adaln=True``), each block executes:

    # Self-attn modulation
    shift_msa, scale_msa, gate_msa = get_ada_factors(slice(0, 3))
    h = rms_norm(x) * (1 + scale_msa) + shift_msa
    x = x + audio_attn1(h, rope=pos_emb) * gate_msa

    # Cross-attn with prompt-AdaLN
    shift_q, scale_q, gate_q = get_ada_factors(slice(6, 9))
    shift_kv, scale_kv = get_prompt_ada_factors()  # 2 vectors from audio_prompt_*
    h = rms_norm(x) * (1 + scale_q) + shift_q
    ctx = a_ctx * (1 + scale_kv) + shift_kv
    x = x + audio_attn2(h, context=ctx) * gate_q

    # FFN modulation
    shift_mlp, scale_mlp, gate_mlp = get_ada_factors(slice(3, 6))
    h = rms_norm(x) * (1 + scale_mlp) + shift_mlp
    x = x + audio_ff(h) * gate_mlp

Per-block saved tables:
- ``audio_scale_shift_table``: [9, hidden] (3 triples for msa, mlp, cross)
- ``audio_prompt_scale_shift_table``: [2, hidden] (shift, scale for context KV)

Saved children (attention/FFN/norms) match the same key paths as the
embeddings connector, so we reuse the LTX primitives from
`dramabox.ltx.*`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..ltx.attention import LTXAttention
from ..ltx.feed_forward import LTXFeedForward
from ..ltx.rms_norm import rms_norm as functional_rms_norm


class LTXBlock(nn.Module):
    """One audio DiT block."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        context_dim: int,
        apply_gated_attention: bool = True,
        cross_attention_adaln: bool = True,
        norm_eps: float = 1e-6,
        rope_type: str = "split",
    ):
        super().__init__()
        self.dim = dim
        self.norm_eps = norm_eps
        self.cross_attention_adaln = cross_attention_adaln

        # Self-attention (queries+keys+values from x)
        self.audio_attn1 = LTXAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            apply_gated_attention=apply_gated_attention,
            rope_type=rope_type,
        )
        # Cross-attention (queries from x, keys+values from context)
        self.audio_attn2 = LTXAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            context_dim=context_dim,
            apply_gated_attention=apply_gated_attention,
            rope_type=rope_type,  # context has no RoPE in our usage
        )
        self.audio_ff = LTXFeedForward(dim, dim_out=dim, mult=4)

        # Per-block AdaLN bias tables (broadcast over batch and tokens)
        ada_coeff = 9 if cross_attention_adaln else 6
        self.audio_scale_shift_table = mx.zeros((ada_coeff, dim), dtype=mx.float32)
        if cross_attention_adaln:
            self.audio_prompt_scale_shift_table = mx.zeros((2, dim), dtype=mx.float32)

    def __call__(
        self,
        x: mx.array,
        *,
        ada_emb: mx.array,
        prompt_ada_emb: mx.array | None,
        context: mx.array,
        rope_cos_sin: tuple[mx.array, mx.array] | None,
        self_attention_mask: mx.array | None = None,
        context_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward.

        Args:
            x: ``[B, T_audio, dim]`` patchified latent.
            ada_emb: ``[B, ada_coeff * dim]`` per-batch AdaLN bias.
            prompt_ada_emb: ``[B, 2 * dim]`` cross-attn context AdaLN bias.
            context: ``[B, T_text, context_dim]`` ``a_ctx`` from the prompt encoder.
            rope_cos_sin: pre-computed RoPE (cos, sin) for the audio sequence.
            self_attention_mask: optional additive mask for audio self-attn.
            context_mask: optional additive mask for cross-attn (None means no mask).
        Returns:
            ``[B, T_audio, dim]``.
        """
        B = x.shape[0]
        dim = self.dim

        # Slice the global ada_emb into 9 per-block factor vectors of shape (B, 1, dim)
        # then add the per-block bias table.
        ada = ada_emb.reshape(B, 1, 9, dim) + self.audio_scale_shift_table[None, None]

        # Self-attention sub-block (factors 0..2)
        shift_msa = ada[:, :, 0, :]
        scale_msa = ada[:, :, 1, :]
        gate_msa = ada[:, :, 2, :]
        h = functional_rms_norm(x, eps=self.norm_eps) * (1 + scale_msa) + shift_msa
        x = x + self.audio_attn1(h, rope_cos_sin=rope_cos_sin, mask=self_attention_mask) * gate_msa

        # Cross-attention sub-block (factors 6..8 for q, prompt-AdaLN for kv)
        shift_q = ada[:, :, 6, :]
        scale_q = ada[:, :, 7, :]
        gate = ada[:, :, 8, :]
        ctx_pa = self.audio_prompt_scale_shift_table[None, None] + prompt_ada_emb.reshape(B, 1, 2, dim)
        shift_kv = ctx_pa[:, :, 0, :]
        scale_kv = ctx_pa[:, :, 1, :]

        attn_input = functional_rms_norm(x, eps=self.norm_eps) * (1 + scale_q) + shift_q
        encoder_hs = context * (1 + scale_kv) + shift_kv
        x = x + self.audio_attn2(attn_input, context=encoder_hs, mask=context_mask) * gate

        # FFN sub-block (factors 3..5)
        shift_mlp = ada[:, :, 3, :]
        scale_mlp = ada[:, :, 4, :]
        gate_mlp = ada[:, :, 5, :]
        h = functional_rms_norm(x, eps=self.norm_eps) * (1 + scale_mlp) + shift_mlp
        x = x + self.audio_ff(h) * gate_mlp

        return x


__all__ = ["LTXBlock"]
