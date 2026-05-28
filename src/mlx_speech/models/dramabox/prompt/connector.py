"""DramaBox `Embeddings1DConnector`.

An 8-block 1D self-attention transformer that processes the audio features
from `FeatureExtractorV2` BEFORE they enter the DiT as cross-attention
context. The standout mechanism is that padded slots in the 1024-token text
sequence are *replaced* by tiled learnable register tokens, after which the
attention mask becomes all-valid (zero additive bias everywhere).

Reference: `.references/DramaBox/ltx2/ltx_core/text_encoders/gemma/embeddings_connector.py:15-198`

Checkpoint contract (per audio-components inspection):

    audio_embeddings_connector.learnable_registers              [128, 2048]
    audio_embeddings_connector.transformer_1d_blocks.{0..7}.
        attn1.{q_norm,k_norm,to_q,to_k,to_v,to_out.0,
              to_gate_logits}.*
        ff.net.0.proj.{weight,bias}
        ff.net.2.{weight,bias}

There is no AdaLN, no scale-shift table, and no cross-attention to the
registers. The "registers" are positional substitutions, not external
context. After the block stack a final functional `rms_norm` is applied
(no learnable params).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from ..ltx.attention import LTXAttention
from ..ltx.feed_forward import LTXFeedForward
from ..ltx.rms_norm import rms_norm as functional_rms_norm
from ..ltx.rope import precompute_split_freqs_1d


class _BasicTransformerBlock1D(nn.Module):
    """One connector block: functional pre-RMSNorm → self-attn → residual →
    functional pre-RMSNorm → FFN → residual.

    NOTE: the pre-norms are *functional* — there are no learnable RMSNorm
    weights on the block level. The only RMSNorm weights inside the block
    are `attn1.q_norm.weight` and `attn1.k_norm.weight` (on the projected
    inner_dim, not the block input).
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        *,
        apply_gated_attention: bool = True,
        rope_type: str = "split",
    ):
        super().__init__()
        self.attn1 = LTXAttention(
            query_dim=dim,
            heads=heads,
            dim_head=dim_head,
            apply_gated_attention=apply_gated_attention,
            rope_type=rope_type,
        )
        self.ff = LTXFeedForward(dim, dim_out=dim)

    def __call__(
        self,
        hidden_states: mx.array,
        *,
        attention_mask: mx.array | None,
        rope_cos_sin: tuple[mx.array, mx.array] | None,
    ) -> mx.array:
        # Self-attention sub-block
        norm = functional_rms_norm(hidden_states)
        attn_out = self.attn1(norm, mask=attention_mask, rope_cos_sin=rope_cos_sin)
        hidden_states = attn_out + hidden_states

        # FFN sub-block
        norm = functional_rms_norm(hidden_states)
        ff_out = self.ff(norm)
        hidden_states = ff_out + hidden_states
        return hidden_states


class Embeddings1DConnector(nn.Module):
    """1D self-attention connector with register-token padding replacement.

    For DramaBox the config is:
        num_attention_heads = 32
        attention_head_dim = 64        # inner_dim = 32 * 64 = 2048
        num_layers = 8
        num_learnable_registers = 128
        positional_embedding_theta = 10000.0
        positional_embedding_max_pos = [4096]
        rope_type = "split"
        double_precision_rope = True (NumPy fp64 grid)
        apply_gated_attention = True

    Forward contract:
        Input:  hidden_states  [B, 1024, 2048]   (audio_feats after rescale)
                attention_mask [B, 1, 1, 1024]   (additive: 0 valid, -large padding)
        Output: hidden_states  [B, 1024, 2048]
                attention_mask [B, 1, 1, 1024]   (zero everywhere; all slots valid)
    """

    def __init__(
        self,
        *,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 8,
        num_learnable_registers: int = 128,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: int = 4096,
        apply_gated_attention: bool = True,
        seq_len: int = 1024,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_layers = num_layers
        self.num_learnable_registers = num_learnable_registers
        self.positional_embedding_theta = positional_embedding_theta
        self.positional_embedding_max_pos = positional_embedding_max_pos
        self.seq_len = seq_len

        # Register parameter: [num_learnable_registers, inner_dim]
        self.learnable_registers = mx.zeros(
            (num_learnable_registers, self.inner_dim), dtype=mx.bfloat16
        )

        # 8 blocks, list-of-modules → serializes as transformer_1d_blocks.0..7
        self.transformer_1d_blocks = [
            _BasicTransformerBlock1D(
                dim=self.inner_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                apply_gated_attention=apply_gated_attention,
                rope_type="split",
            )
            for _ in range(num_layers)
        ]

    # ----- register replacement -------------------------------------------

    def _replace_padded_with_learnable_registers(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Replace padded slots with tiled learnable registers.

        Upstream logic (`embeddings_connector.py:135-161`):

            1. Tile registers `seq_len / num_registers` times → [seq_len, inner_dim]
            2. Recover binary mask from the additive mask: 1 = valid (additive ≥ -9000),
               0 = pad
            3. Pack non-padded values to the front of the sequence, zero-pad the
               tail to seq_len
            4. Flip the binary mask along seq dim → puts padded slots at the back
            5. Output = flipped_mask * adjusted + (1 - flipped_mask) * registers
               (non-padded positions keep features, padded positions get registers)
            6. Return all-zero additive mask (every slot is valid)

        For DramaBox left-padded input (pad at positions 0..k, real tokens at
        k+1..end), the flipped binary mask is `[1, 1, ..., 1, 0, ..., 0]` —
        which after the multiply puts the packed real tokens at the front and
        the registers at the back. That is, the connector output has a
        deterministic "real-then-registers" ordering regardless of input
        padding side.
        """
        B, T, D = hidden_states.shape
        assert T % self.num_learnable_registers == 0, (
            f"seq_len {T} must be divisible by num_learnable_registers "
            f"{self.num_learnable_registers}"
        )
        num_dup = T // self.num_learnable_registers
        # Tile registers along the sequence dim: [num_dup * num_registers, D]
        # Use repeat along axis=0 with `num_dup` copies. `tile` semantics:
        # tile(a, (n, 1)) = repeat the array n times along axis 0.
        tiled_registers = mx.tile(self.learnable_registers, (num_dup, 1))  # [T, D]
        tiled_registers = mx.broadcast_to(tiled_registers, (B, T, D))

        # Recover binary mask. The connector receives a [B, 1, 1, T] additive
        # mask in compute dtype. A "valid" position is one whose additive
        # bias is essentially zero (> -9000, matching the upstream
        # heuristic). Reshape to [B, T] for sequence-axis operations.
        mask_bt = (attention_mask.reshape(B, T) >= -9000.0).astype(mx.int32)  # [B, T]

        # Sort each row so all "valid" positions (mask=1) come first.
        # We need a stable sort that preserves the relative order of valid
        # tokens. We use `argsort` on `-mask_bt` (so 1s sort before 0s) but
        # that isn't stable in MLX. Easier: gather valid tokens to the front
        # using a cumulative-sum-based scatter.
        adjusted = self._pack_valid_to_front(hidden_states, mask_bt)  # [B, T, D]

        # Flip the binary mask along the seq dim → puts padded slots at the
        # back, valid slots at the front. After flip, mask_flipped[t] is 1
        # for positions that originally were valid (anywhere) — but since
        # we already packed valid to the front and the mask had `n_valid`
        # ones, the flipped mask has the FIRST n_valid positions as 1 only
        # if originally the LAST n_valid were 1. For left-padding, originally
        # `[0,0,...,0,1,...,1]`; flipped is `[1,...,1,0,...,0]`. Which is
        # exactly the "valid at the front, pad at back" pattern matching
        # our packed `adjusted`.
        flipped = mask_bt[:, ::-1]  # [B, T] — flip along the sequence axis
        flipped_3d = flipped.reshape(B, T, 1).astype(hidden_states.dtype)

        out = flipped_3d * adjusted + (1 - flipped_3d) * tiled_registers.astype(hidden_states.dtype)

        # New attention mask: all-zero additive (i.e., every slot is valid)
        new_mask = mx.zeros_like(attention_mask)
        return out, new_mask

    @staticmethod
    def _pack_valid_to_front(
        hidden_states: mx.array,
        binary_mask: mx.array,
    ) -> mx.array:
        """Pack valid tokens (mask=1) to the front of the sequence.

        We compute a target index per token: ``target = cumsum(mask) - 1`` for
        valid positions, and an out-of-range index for padding (so the
        scatter dumps them into a discarded slot). Then ``scatter`` along the
        seq axis builds the packed tensor.

        Implementation detail: MLX doesn't have a direct scatter-by-index
        op for arbitrary integer indices, but we can use the equivalent
        ``mx.take_along_axis`` / argsort trick. Since mask values are 0/1,
        stable-sorting by ``-mask`` puts 1s before 0s with a stable order
        for ties.
        """
        # Use a stable sort key: rows where mask=1 should come first.
        # MLX `argsort` is not guaranteed stable. We bake in token-index as
        # a tiebreaker: sort key = mask * (T + 1) - position_index ensures
        # all mask=1 rows sort above mask=0 rows AND preserves the original
        # left-to-right order among mask=1 rows.
        B, T = binary_mask.shape
        position = mx.arange(T)[None, :].astype(mx.int32)  # [1, T]
        # sort key: large for mask=1 in original order, small for mask=0.
        # Use: key = mask * (T+1) - position. Then descending sort.
        key = binary_mask * (T + 1) - position  # [B, T]
        # MLX `argsort` returns ascending order; we want descending, so sort
        # by -key.
        order = mx.argsort(-key, axis=1)  # [B, T] indices into original

        # Gather hidden_states by `order` along the seq axis.
        packed = mx.take_along_axis(
            hidden_states,
            order[:, :, None].astype(mx.int32),
            axis=1,
        )
        # Zero out positions beyond the original number of valid tokens.
        # `num_valid = sum(mask)` per batch row. positions ≥ num_valid get zero.
        num_valid = mx.sum(binary_mask, axis=1, keepdims=True)  # [B, 1]
        valid_mask = (mx.arange(T)[None, :] < num_valid).astype(packed.dtype)  # [B, T]
        packed = packed * valid_mask[:, :, None]
        return packed

    # ----- forward --------------------------------------------------------

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Forward.

        Args:
            hidden_states: ``[B, seq_len, inner_dim]`` audio features.
            attention_mask: ``[B, 1, 1, seq_len]`` additive bias.

        Returns:
            ``(out, new_attention_mask)`` — both with the same shapes as
            inputs. `new_attention_mask` is all-zero (all slots valid).
        """
        if self.num_learnable_registers:
            hidden_states, attention_mask = self._replace_padded_with_learnable_registers(
                hidden_states, attention_mask
            )

        # Pre-compute RoPE for the sequence length once. The connector uses
        # 1D split RoPE with a NumPy fp64 inverse-frequency grid.
        T = hidden_states.shape[1]
        rope_cos_sin = precompute_split_freqs_1d(
            seq_len=T,
            inner_dim=self.inner_dim,
            num_heads=self.num_attention_heads,
            theta=self.positional_embedding_theta,
            max_pos=self.positional_embedding_max_pos,
            out_dtype=hidden_states.dtype,
        )

        for block in self.transformer_1d_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
            )

        # Final functional RMSNorm (no learnable params)
        hidden_states = functional_rms_norm(hidden_states)
        return hidden_states, attention_mask


__all__ = ["Embeddings1DConnector"]
