"""LTX audio-only DiT — top-level container.

Forward path:

    x [B, T_audio, 128]          # patchified latent
    a_ctx [B, T_text, 2048]      # prompt encoder output
    sigma [B]                    # current diffusion sigma

        x = audio_patchify_proj(x)              # [B, T, 2048]
        ada_emb, t_emb = audio_adaln_single(sigma * 1000)
        prompt_ada, _ = audio_prompt_adaln_single(sigma * 1000)
        rope_cs = precompute_split_rope_for_audio_positions(...)

        for block in transformer_blocks:
            x = block(x, ada_emb=ada_emb, prompt_ada_emb=prompt_ada,
                      context=a_ctx, rope_cos_sin=rope_cs)

        final_shift, final_scale = (audio_scale_shift_table + t_emb).split
        x = layer_norm(x) * (1 + final_scale) + final_shift
        velocity = audio_proj_out(x)         # [B, T, 128]

The forward returns velocity. `X0Model` (Stage 7) wraps this to compute
denoised x0 = noisy - velocity * sigma.

Reference: `.references/DramaBox/ltx2/ltx_core/model/transformer/model.py:31-430`
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..ltx.rope import precompute_split_freqs_from_positions
from .block import LTXBlock
from .config import DiTConfig
from .timestep import AdaLayerNormSingle


class LTXModel(nn.Module):
    """The 48-block audio DiT, returning velocity."""

    def __init__(self, config: DiTConfig):
        super().__init__()
        self.config = config
        hidden = config.audio_inner_dim

        self.audio_patchify_proj = nn.Linear(config.audio_in_channels, hidden, bias=True)

        # AdaLN modulators
        ada_coeff = 9 if config.cross_attention_adaln else 6
        self.audio_adaln_single = AdaLayerNormSingle(hidden, coeff=ada_coeff)
        if config.cross_attention_adaln:
            self.audio_prompt_adaln_single = AdaLayerNormSingle(hidden, coeff=2)
        else:
            self.audio_prompt_adaln_single = None

        # 48 transformer blocks
        self.transformer_blocks = [
            LTXBlock(
                dim=hidden,
                heads=config.audio_num_attention_heads,
                dim_head=config.audio_attention_head_dim,
                context_dim=config.audio_cross_attention_dim,
                apply_gated_attention=config.apply_gated_attention,
                cross_attention_adaln=config.cross_attention_adaln,
                norm_eps=config.norm_eps,
                rope_type=config.rope_type,
            )
            for _ in range(config.num_layers)
        ]

        # Output head
        self.audio_scale_shift_table = mx.zeros((2, hidden), dtype=mx.float32)
        # Upstream uses `nn.LayerNorm(elementwise_affine=False)` — i.e. no weight/bias.
        # We implement it inline since MLX has no `elementwise_affine` flag.
        self.audio_proj_out = nn.Linear(hidden, config.audio_out_channels, bias=True)

    def _norm_out(self, x: mx.array) -> mx.array:
        """LayerNorm without learnable params (matches PyTorch
        `nn.LayerNorm(elementwise_affine=False)`)."""
        orig_dtype = x.dtype
        x32 = x.astype(mx.float32)
        mean = mx.mean(x32, axis=-1, keepdims=True)
        var = mx.mean((x32 - mean) ** 2, axis=-1, keepdims=True)
        out = (x32 - mean) * mx.rsqrt(var + self.config.norm_eps)
        return out.astype(orig_dtype)

    def __call__(
        self,
        x: mx.array,
        *,
        a_ctx: mx.array,
        sigma: mx.array,
        positions: mx.array | None = None,
        rope_cos_sin: tuple[mx.array, mx.array] | None = None,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """Forward returning velocity ``[B, T_audio, audio_out_channels=128]``.

        Args:
            x: patchified latent ``[B, T_audio, 128]``.
            a_ctx: prompt encoder output ``[B, T_text, 2048]``.
            sigma: per-batch ``[B]`` diffusion sigma. Scaled by
                ``timestep_scale_multiplier=1000`` before AdaLN.
            positions: optional ``[B, 1, T, 2]`` (start_s, end_s) timing pairs
                from the patchifier — what the reference uses for RoPE. When
                provided this drives the RoPE computation (mirroring
                `transformer_args.py:166-173`).
            rope_cos_sin: optional pre-computed RoPE; takes precedence over
                ``positions``. Used by callers that batch multiple guidance
                passes through the DiT and want to share the RoPE table.
            attention_mask: optional additive self-attention mask broadcastable
                to ``[B, heads, T_audio, T_audio]``.

        Returns:
            velocity ``[B, T_audio, 128]``.
        """
        B, T, _ = x.shape
        compute_dtype = x.dtype

        # Input projection
        x = self.audio_patchify_proj(x)  # [B, T, hidden]

        # RoPE for the audio sequence — prefer explicit cos/sin, then positions.
        if rope_cos_sin is None:
            if positions is None:
                raise ValueError(
                    "LTXModel forward requires either `rope_cos_sin` or `positions` "
                    "(the patchifier's [B, 1, T, 2] start/end timings)."
                )
            rope_cos_sin = precompute_split_freqs_from_positions(
                positions=positions,
                inner_dim=self.config.audio_inner_dim,
                num_heads=self.config.audio_num_attention_heads,
                theta=self.config.positional_embedding_theta,
                max_pos=self.config.audio_positional_embedding_max_pos[0],
                out_dtype=compute_dtype,
            )

        # Scaled timestep
        sigma_scaled = sigma.astype(mx.float32) * float(self.config.timestep_scale_multiplier)

        ada_emb, embedded_t = self.audio_adaln_single(sigma_scaled, compute_dtype)  # [B, 9*hidden], [B, hidden]
        if self.audio_prompt_adaln_single is not None:
            prompt_ada, _ = self.audio_prompt_adaln_single(sigma_scaled, compute_dtype)
        else:
            prompt_ada = None

        # Block stack
        for block in self.transformer_blocks:
            x = block(
                x,
                ada_emb=ada_emb,
                prompt_ada_emb=prompt_ada,
                context=a_ctx,
                rope_cos_sin=rope_cos_sin,
                self_attention_mask=attention_mask,
            )

        # Final AdaLN + output projection
        ada_out = self.audio_scale_shift_table[None, None] + embedded_t.reshape(B, 1, 1, -1)
        # Upstream: shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]
        # But the bias table has shape (2, hidden), and the reshape is (B, 1, 1, hidden).
        # Need to broadcast properly: bias is (1, 1, 2, hidden), emb is (B, 1, 1, hidden).
        # Add via broadcasting: (B, 1, 2, hidden).
        bias = self.audio_scale_shift_table[None, None]  # (1, 1, 2, hidden)
        embedded = embedded_t.reshape(B, 1, 1, -1)  # (B, 1, 1, hidden) broadcasts to (B, 1, 2, hidden)
        scale_shift = bias + embedded  # (B, 1, 2, hidden)
        shift_final = scale_shift[:, :, 0, :]
        scale_final = scale_shift[:, :, 1, :]
        x = self._norm_out(x)
        x = x * (1 + scale_final) + shift_final
        velocity = self.audio_proj_out(x)
        return velocity


__all__ = ["LTXModel"]
