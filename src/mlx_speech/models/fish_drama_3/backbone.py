"""Drama 3 text backbone: token embeddings, hybrid decoder layers, final norm, lm_head."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .._cache import BoundedKVCache
from .config import ExpertQuantization, FishDrama3TextConfig
from .layers import DecoderLayer, DeltaNetState, ZeroCenteredRMSNorm

LayerCache = BoundedKVCache | DeltaNetState


class FishDrama3Backbone(nn.Module):
    def __init__(self, config: FishDrama3TextConfig, quantization: ExpertQuantization):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            DecoderLayer(config, index, quantization)
            for index in range(config.num_hidden_layers)
        ]
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def make_cache(
        self, *, capacity: int, batch_size: int = 1, dtype: mx.Dtype = mx.bfloat16
    ) -> list[LayerCache]:
        """Per-layer state for capacity rows: K/V for gated attention, fixed-size
        recurrent state for Gated DeltaNet."""
        return [
            DeltaNetState.allocate(self.config, batch_size=batch_size, dtype=dtype)
            if layer.is_linear
            else BoundedKVCache.allocate(
                batch_size=batch_size,
                capacity=capacity,
                num_heads=self.config.num_key_value_heads,
                head_dim=self.config.head_dim,
                key_dtype=dtype,
                value_dtype=dtype,
            )
            for layer in self.layers
        ]

    def __call__(self, inputs_embeds: mx.array, cache: list[LayerCache]) -> mx.array:
        """Final-norm hidden states [batch, rows, hidden] of rows that continue the
        cached sequence."""
        hidden = inputs_embeds
        for layer, layer_cache in zip(self.layers, cache, strict=True):
            hidden = layer(hidden, layer_cache)
        return self.norm(hidden)

    def logits(self, hidden: mx.array, token_ids: mx.array) -> mx.array:
        """Logits of token_ids only, [..., len(token_ids)]: sampling never looks
        past eos and the semantic band, so the other 248k rows are never read."""
        return hidden @ self.lm_head.weight[token_ids].T
