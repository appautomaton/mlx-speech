"""Drama 3 audio decoder: a codebook tower that completes each frame's codes.

A frame row is [main token id, code 0, ..., code N-1]; code 0 is the main token's
offset in the semantic band. The backbone samples the main token; the tower attends
over earlier frames and predicts codes 1..N-1, one codebook per residual layer.
"""

from __future__ import annotations

import math
from functools import cached_property

import mlx.core as mx
import mlx.nn as nn

from .._cache import BoundedKVCache
from .config import FishDrama3AudioDecoderConfig, FishDrama3Config
from .layers import ZeroCenteredRMSNorm, padded_mask, sigmoid_gate, swiglu
from .sampling import (
    MAIN_EPSILON_CUTOFF,
    RESIDUAL_CODEBOOK_SIZE,
    reference_sample,
    replace_repeats,
)


@mx.compile
def code_embedding(
    table: mx.array, codes: mx.array, semantic: mx.array, like: mx.array
) -> mx.array:
    """Embeddings of codes on semantic rows, zero elsewhere, in like's dtype."""
    embedded = mx.take(table, mx.clip(codes, 0, table.shape[0] - 1), axis=0)
    return mx.where(semantic[:, None], embedded, 0.0).astype(like.dtype)


class TowerAttention(nn.Module):
    """Gated attention over frames with interleaved partial RoPE; wqkv rows are
    [Q | Gate | K | V]."""

    def __init__(self, config: FishDrama3AudioDecoderConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_local_heads = config.n_local_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.rope_base = config.rope_base
        self.q_width = config.n_head * config.head_dim
        self.kv_width = config.n_local_heads * config.head_dim
        self.wqkv = nn.Linear(
            config.dim, 2 * self.q_width + 2 * self.kv_width, bias=False
        )
        self.wo = nn.Linear(self.q_width, config.dim, bias=False)
        # Always on, whatever audio_decoder_config.attention_qk_norm says: training
        # used q/k norm, but the upstream exporter never writes that field.
        self.q_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(config.head_dim, config.norm_eps)

    def rope(self, x: mx.array, offset: int | mx.array) -> mx.array:
        return mx.fast.rope(
            x,
            self.rotary_dim,
            traditional=True,
            base=self.rope_base,
            scale=1.0,
            offset=offset,
        )

    def __call__(
        self, x: mx.array, cache: BoundedKVCache, pads: mx.array | None = None
    ) -> mx.array:
        """x [rows, dim]: without pads, one sequence of rows frames that follows
        the cache; with pads, one frame for each of rows batched sequences, whose
        cached frames start at slot pads[row]."""
        batch, length = (1, x.shape[0]) if pads is None else (x.shape[0], 1)
        offset = cache.offset if pads is None else cache.offset - pads
        query, gate, key, value = mx.split(
            self.wqkv(x),
            [self.q_width, 2 * self.q_width, 2 * self.q_width + self.kv_width],
            axis=-1,
        )
        query = self.q_norm(query.reshape(batch, length, self.n_head, self.head_dim))
        key = self.k_norm(
            key.reshape(batch, length, self.n_local_heads, self.head_dim)
        )
        value = value.reshape(batch, length, self.n_local_heads, self.head_dim)
        query = self.rope(query.transpose(0, 2, 1, 3), offset)
        key = self.rope(key.transpose(0, 2, 1, 3), offset)
        cache.append(key.transpose(0, 2, 1, 3), value)
        keys, values = cache.fetch()
        attended = mx.fast.scaled_dot_product_attention(
            query,
            keys.transpose(0, 2, 1, 3),
            values.transpose(0, 2, 1, 3),
            scale=self.head_dim**-0.5,
            mask=padded_mask(pads, length, keys.shape[1]),
        )
        attended = attended.transpose(0, 2, 1, 3).reshape(
            batch * length, self.q_width
        )
        return self.wo(sigmoid_gate(attended, gate))


class TowerMLP(nn.Module):
    def __init__(self, config: FishDrama3AudioDecoderConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(swiglu(self.w1(x), self.w3(x)))


class TowerTransformerLayer(nn.Module):
    def __init__(self, config: FishDrama3AudioDecoderConfig):
        super().__init__()
        self.attention = TowerAttention(config)
        self.attention_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.feed_forward = TowerMLP(config)

    def __call__(
        self, x: mx.array, cache: BoundedKVCache, pads: mx.array | None = None
    ) -> mx.array:
        x = x + self.attention(self.attention_norm(x), cache, pads)
        return x + self.feed_forward(self.ffn_norm(x))


class TowerLayer(nn.Module):
    """Mix one code's embedding into the hidden state, then one pre-norm block."""

    def __init__(self, config: FishDrama3AudioDecoderConfig, *, shared: bool):
        super().__init__()
        self.shared = shared
        # The checkpoint names the embedding differently in shared and residual layers.
        embedding = nn.Embedding(config.vocab_size, config.dim)
        if shared:
            self.semantic_embedding = embedding
        else:
            self.codebook_embedding = embedding
        self.enorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.hnorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)
        self.eh_proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.transformer_layer = TowerTransformerLayer(config)
        self.final_layernorm = ZeroCenteredRMSNorm(config.dim, config.norm_eps)

    def __call__(
        self,
        hidden: mx.array,
        codes: mx.array,
        semantic: mx.array,
        cache: BoundedKVCache,
        shared_input: mx.array | None = None,
        pads: mx.array | None = None,
    ) -> mx.array:
        embedding = self.semantic_embedding if self.shared else self.codebook_embedding
        embedded = code_embedding(embedding.weight, codes, semantic, hidden)
        if shared_input is not None:
            embedded = embedded + shared_input
        mixed = self.eh_proj(
            mx.concatenate([self.enorm(embedded), self.hnorm(hidden)], axis=-1)
        )
        return self.final_layernorm(self.transformer_layer(mixed, cache, pads))


class FishDrama3AudioDecoder(nn.Module):
    def __init__(self, config: FishDrama3Config):
        super().__init__()
        decoder = config.audio_decoder
        self.config = decoder
        self.semantic_begin = config.semantic_start_token_id
        self.semantic_end = config.semantic_end_token_id
        self.eos_id = config.eos_token_id
        self.codebook_embeddings = nn.Embedding(
            decoder.vocab_size * decoder.num_codebooks, decoder.dim
        )
        self.shared_layers = [
            TowerLayer(decoder, shared=True)
            for _ in range(decoder.shared_enrichment_layers)
        ]
        self.residual_layers = [
            TowerLayer(decoder, shared=False) for _ in range(decoder.num_codebooks - 1)
        ]
        self.codebook_heads = [
            nn.Linear(decoder.dim, decoder.vocab_size, bias=False)
            for _ in range(decoder.num_codebooks - 1)
        ]

    @cached_property
    def main_candidates(self) -> mx.array:
        """Token ids the main token is drawn from, ascending: eos and the band."""
        band = mx.arange(self.semantic_begin, self.semantic_end + 1)
        return mx.sort(mx.concatenate([mx.array([self.eos_id]), band]))

    def make_cache(
        self, *, capacity: int, dtype: mx.Dtype = mx.bfloat16
    ) -> list[BoundedKVCache]:
        """One K/V cache over frame positions per shared, then residual, layer."""
        return [
            BoundedKVCache.allocate(
                batch_size=1,
                capacity=capacity,
                num_heads=self.config.n_local_heads,
                head_dim=self.config.head_dim,
                key_dtype=dtype,
                value_dtype=dtype,
            )
            for _ in range(len(self.shared_layers) + len(self.residual_layers))
        ]

    def is_semantic(self, tokens: mx.array) -> mx.array:
        return (tokens >= self.semantic_begin) & (tokens <= self.semantic_end)

    def embedding_sum(self, frames: mx.array, first: int, end: int) -> mx.array:
        """Summed embeddings of codes first..end-1 of each frame row."""
        if first == end:
            return mx.zeros(
                (frames.shape[0], self.config.dim),
                dtype=self.codebook_embeddings.weight.dtype,
            )
        offsets = mx.arange(first, end) * self.config.vocab_size
        return self.codebook_embeddings(frames[:, 1 + first : 1 + end] + offsets).sum(
            axis=1
        )

    def embed_backbone(self, frames: mx.array, text_embeddings: mx.array) -> mx.array:
        """Backbone input: semantic rows add the embeddings of codes 1..K."""
        count = self.config.vq_backbone_codebooks
        extra = self.embedding_sum(frames, 1, 1 + count)
        mixed = (text_embeddings + extra) / math.sqrt(1 + count)
        semantic = self.is_semantic(frames[:, 0])
        return mx.where(semantic[:, None], mixed, text_embeddings)

    def run_shared(
        self,
        hidden: mx.array,
        c0: mx.array,
        semantic: mx.array,
        frames: mx.array,
        cache: list[BoundedKVCache],
        pads: mx.array | None = None,
    ) -> mx.array:
        # The first shared layer also sees the codes of the input row that the
        # backbone did not.
        unseen = self.embedding_sum(
            frames, 1 + self.config.vq_backbone_codebooks, self.config.num_codebooks
        )
        unseen = mx.where(self.is_semantic(frames[:, 0])[:, None], unseen, 0.0)
        for index, layer in enumerate(self.shared_layers):
            hidden = layer(
                hidden,
                c0,
                semantic,
                cache[index],
                unseen if index == 0 else None,
                pads,
            )
        return hidden

    def prefill(
        self,
        hidden: mx.array,
        frames: mx.array,
        next_frames: mx.array,
        cache: list[BoundedKVCache],
    ) -> None:
        """Teacher-force the caches: row p is labelled with frame p + 1."""
        semantic = self.is_semantic(next_frames[:, 0])
        hidden = self.run_shared(hidden, next_frames[:, 1], semantic, frames, cache)
        shared = len(self.shared_layers)
        for index, layer in enumerate(self.residual_layers):
            hidden = layer(
                hidden, next_frames[:, index + 1], semantic, cache[shared + index]
            )

    def decode_step(
        self,
        main_logits: mx.array,
        hidden: mx.array,
        frame: mx.array,
        history: mx.array,
        cache: list[BoundedKVCache],
        *,
        temperature: float,
        top_k: int,
        top_p: float,
        pads: mx.array | None = None,
        keys: mx.array | None = None,
    ) -> mx.array:
        """Sample the frame after the current row of each sequence under reference
        sampling (see sampling): [rows, 1 + num_codebooks].

        main_logits [rows, len(main_candidates)] are the backbone logits of the
        candidates; hidden [rows, dim], frame [rows, 1 + N] and history
        [rows, frames] belong to each sequence's current row. One sequence needs no
        pads; batched ones pass theirs (see TowerAttention). keys [rows,
        num_codebooks] draw each sequence's codes; None draws from the global
        generator.
        """
        candidates = self.main_candidates
        draw = [None] * self.config.num_codebooks if keys is None else keys.swapaxes(0, 1)
        first = candidates[
            reference_sample(
                main_logits,
                temperature,
                top_k,
                top_p,
                epsilon_cutoff=MAIN_EPSILON_CUTOFF,
                keys=draw[0],
            )
        ]
        best = candidates[mx.argmax(main_logits, axis=-1)]
        main = replace_repeats(first, best, history)
        c0 = mx.clip(main - self.semantic_begin, 0, self.config.vocab_size - 1)
        semantic = self.is_semantic(main)
        hidden = self.run_shared(hidden, c0, semantic, frame, cache, pads)
        codes = [c0]
        shared = len(self.shared_layers)
        for index, (layer, head) in enumerate(
            zip(self.residual_layers, self.codebook_heads, strict=True)
        ):
            hidden = layer(
                hidden, codes[-1], semantic, cache[shared + index], pads=pads
            )
            codes.append(
                reference_sample(
                    head(hidden),
                    temperature,
                    top_k,
                    top_p,
                    legal_columns=RESIDUAL_CODEBOOK_SIZE,
                    keys=draw[index + 1],
                )
            )
        return mx.stack([main, *codes], axis=-1).astype(mx.int32)
