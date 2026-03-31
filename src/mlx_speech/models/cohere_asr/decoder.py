"""CohereAsr Transformer decoder in pure MLX."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import CohereAsrDecoderConfig


# ---------------------------------------------------------------------------
# Self-attention (causal, no RoPE, learned position embeddings used externally)
# ---------------------------------------------------------------------------

class CohereAsrSelfAttention(nn.Module):
    def __init__(self, config: CohereAsrDecoderConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        bias = config.attention_bias

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)

        self._layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        causal_mask: mx.array | None = None,
        kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Args:
            x:          (batch, T, hidden)
            causal_mask: additive mask (batch, 1, T, T_total) or None
            kv_cache:   (k_past, v_past) each (batch, heads, T_past, head_dim)

        Returns:
            output: (batch, T, hidden)
            new_kv: (k, v) each (batch, heads, T_total, head_dim)
        """
        batch, T, _ = x.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(batch, T, h, d).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch, T, h, d).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch, T, h, d).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = mx.concatenate([k_past, k], axis=2)
            v = mx.concatenate([v_past, v], axis=2)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, h, T, T_total)
        if causal_mask is not None:
            scores = scores + causal_mask

        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, T, -1)
        return self.o_proj(out), (k, v)


# ---------------------------------------------------------------------------
# Cross-attention (non-causal; encoder KV cached after first step)
# ---------------------------------------------------------------------------

class CohereAsrCrossAttention(nn.Module):
    def __init__(self, config: CohereAsrDecoderConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = config.head_dim ** -0.5
        bias = config.attention_bias

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)

        self._layer_idx = layer_idx

    def __call__(
        self,
        x: mx.array,
        encoder_states: mx.array,
        cross_kv_cache: tuple[mx.array, mx.array] | None = None,
        encoder_mask: mx.array | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        """
        Args:
            x:               (batch, T_dec, hidden)
            encoder_states:  (batch, T_enc, hidden)  — already projected by decoder.proj
            cross_kv_cache:  cached (k, v) from encoder; once set, encoder_states is ignored
            encoder_mask:    (batch, 1, 1, T_enc) additive mask or None

        Returns:
            output: (batch, T_dec, hidden)
            cross_kv: (k, v) — encoder K/V (reusable across decode steps)
        """
        batch, T_dec, _ = x.shape
        h, d = self.num_heads, self.head_dim

        q = self.q_proj(x).reshape(batch, T_dec, h, d).transpose(0, 2, 1, 3)

        if cross_kv_cache is not None:
            k, v = cross_kv_cache
        else:
            T_enc = encoder_states.shape[1]
            k = self.k_proj(encoder_states).reshape(batch, T_enc, h, d).transpose(0, 2, 1, 3)
            v = self.v_proj(encoder_states).reshape(batch, T_enc, h, d).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, h, T_dec, T_enc)
        if encoder_mask is not None:
            scores = scores + encoder_mask

        weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, T_dec, -1)
        return self.o_proj(out), (k, v)


# ---------------------------------------------------------------------------
# Decoder MLP (two-layer with relu)
# ---------------------------------------------------------------------------

class CohereAsrMLP(nn.Module):
    def __init__(self, config: CohereAsrDecoderConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.relu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class CohereAsrDecoderLayer(nn.Module):
    def __init__(self, config: CohereAsrDecoderConfig, layer_idx: int):
        super().__init__()
        self.self_attn = CohereAsrSelfAttention(config, layer_idx)
        self.encoder_attn = CohereAsrCrossAttention(config, layer_idx)
        self.mlp = CohereAsrMLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.final_layernorm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        x: mx.array,
        encoder_states: mx.array,
        causal_mask: mx.array | None = None,
        encoder_mask: mx.array | None = None,
        self_kv_cache: tuple[mx.array, mx.array] | None = None,
        cross_kv_cache: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array], tuple[mx.array, mx.array]]:
        # Self-attention
        residual = x
        x_norm = self.input_layernorm(x)
        attn_out, new_self_kv = self.self_attn(x_norm, causal_mask, self_kv_cache)
        x = residual + attn_out

        # Cross-attention
        residual = x
        x_norm = self.post_attention_layernorm(x)
        cross_out, new_cross_kv = self.encoder_attn(
            x_norm, encoder_states, cross_kv_cache, encoder_mask
        )
        x = residual + cross_out

        # MLP
        residual = x
        x = residual + self.mlp(self.final_layernorm(x))

        return x, new_self_kv, new_cross_kv


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class CohereAsrDecoder(nn.Module):
    def __init__(self, config: CohereAsrDecoderConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.embedding_layernorm = nn.LayerNorm(config.hidden_size)
        # Project encoder hidden states to decoder hidden size
        self.proj = nn.Linear(config.encoder_hidden_size, config.hidden_size, bias=True)
        self.layers = [CohereAsrDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.norm = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        input_ids: mx.array,
        encoder_states: mx.array,
        encoder_mask: mx.array | None = None,
        self_kv_caches: list[tuple[mx.array, mx.array]] | None = None,
        cross_kv_caches: list[tuple[mx.array, mx.array]] | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, list[tuple], list[tuple]]:
        """
        Args:
            input_ids:      (batch, T) int32
            encoder_states: (batch, T_enc, encoder_hidden) — raw encoder output
            encoder_mask:   (batch, T_enc) bool or None
            self_kv_caches: per-layer self-attn KV cache (or None for fresh)
            cross_kv_caches: per-layer cross-attn KV cache (or None to compute)
            position_offset: offset for position IDs (used during decode steps)

        Returns:
            hidden: (batch, T, hidden_size)
            new_self_kv_caches:  list of (k,v) per layer
            new_cross_kv_caches: list of (k,v) per layer
        """
        batch, T = input_ids.shape

        # Embed tokens + position
        token_emb = self.embed_tokens(input_ids)
        positions = mx.arange(position_offset, position_offset + T, dtype=mx.int32)
        pos = self.pos_emb(positions)
        x = self.embedding_layernorm(token_emb + pos)

        # Project encoder states once
        enc = self.proj(encoder_states)  # (batch, T_enc, hidden)

        # Build encoder additive mask
        enc_bias: mx.array | None = None
        if encoder_mask is not None:
            neg_inf = mx.array(mx.finfo(mx.float32).min, dtype=enc.dtype)
            # (batch, 1, 1, T_enc)
            enc_bias = mx.where(
                encoder_mask[:, None, None, :],
                mx.zeros((1,), dtype=enc.dtype),
                neg_inf,
            )

        # Build causal mask for self-attention
        T_past = self_kv_caches[0][0].shape[2] if self_kv_caches else 0
        T_total = T_past + T
        causal = nn.MultiHeadAttention.create_additive_causal_mask(T_total, dtype=x.dtype)
        # We only need the rows corresponding to the current T tokens
        causal_mask = causal[None, None, T_past:, :]  # (1, 1, T, T_total)

        new_self_kvs: list[tuple] = []
        new_cross_kvs: list[tuple] = []

        for i, layer in enumerate(self.layers):
            skv = self_kv_caches[i] if self_kv_caches else None
            ckv = cross_kv_caches[i] if cross_kv_caches else None
            x, new_skv, new_ckv = layer(
                x, enc, causal_mask, enc_bias, skv, ckv
            )
            new_self_kvs.append(new_skv)
            new_cross_kvs.append(new_ckv)

        x = self.norm(x)
        return x, new_self_kvs, new_cross_kvs


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class CohereAsrForConditionalGeneration(nn.Module):
    """CohereAsr encoder-decoder model with LM head."""

    def __init__(self, config: "CohereAsrConfig"):  # noqa: F821 — forward ref
        super().__init__()
        from .encoder import ParakeetEncoder

        self.encoder = ParakeetEncoder(config.encoder)
        self.decoder = CohereAsrDecoder(config.decoder)
        # proj_out weight is tied to embed_tokens.weight
        self.proj_out = nn.Linear(config.decoder.hidden_size, config.decoder.vocab_size, bias=True)

    def encode(
        self,
        features: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """Run encoder only. Returns (encoder_states, output_mask)."""
        return self.encoder(features, attention_mask)

    def decode_step(
        self,
        input_ids: mx.array,
        encoder_states: mx.array,
        encoder_mask: mx.array | None = None,
        self_kv_caches: list | None = None,
        cross_kv_caches: list | None = None,
        position_offset: int = 0,
    ) -> tuple[mx.array, list, list]:
        """Single decode step (or prefill).

        Returns:
            logits: (batch, T, vocab_size)
            new_self_kv_caches
            new_cross_kv_caches
        """
        hidden, new_self_kvs, new_cross_kvs = self.decoder(
            input_ids,
            encoder_states,
            encoder_mask,
            self_kv_caches,
            cross_kv_caches,
            position_offset,
        )
        logits = self.proj_out(hidden)
        return logits, new_self_kvs, new_cross_kvs
