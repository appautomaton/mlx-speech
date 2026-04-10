from __future__ import annotations

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn

from .cache import KVCache
from .config import FishS2ProConfig
from .layers import FeedForward, FishRotaryEmbedding, TransformerBlock

__all__ = [
    "DualARTransformer",
    "FeedForward",
    "FishRotaryEmbedding",
    "ForwardResult",
    "TransformerBlock",
]


@dataclass
class ForwardResult:
    logits: mx.array
    hidden_states: mx.array


class DualARTransformer(nn.Module):
    def __init__(self, config: FishS2ProConfig):
        super().__init__()
        self.config = config
        tc = config.text_config
        ac = config.audio_decoder_config
        self.embeddings = nn.Embedding(tc.vocab_size, tc.dim)
        self.codebook_embeddings = nn.Embedding(
            ac.vocab_size * ac.num_codebooks, tc.dim
        )
        self.layers = [TransformerBlock.from_text_config(tc) for _ in range(tc.n_layer)]
        self.norm = nn.RMSNorm(tc.dim, eps=tc.norm_eps)
        self.fast_project_in = (
            nn.Linear(tc.dim, ac.dim, bias=False) if tc.dim != ac.dim else nn.Identity()
        )
        self.fast_embeddings = nn.Embedding(ac.vocab_size, ac.dim)
        self.fast_layers = [
            TransformerBlock.from_audio_config(ac) for _ in range(ac.n_layer)
        ]
        self.fast_norm = nn.RMSNorm(ac.dim, eps=ac.norm_eps)
        self.fast_output = nn.Linear(ac.dim, ac.vocab_size, bias=False)

    @property
    def num_codebooks(self) -> int:
        return self.config.audio_decoder_config.num_codebooks

    def _validate_input_shape(self, inp: mx.array) -> None:
        expected_rows = self.num_codebooks + 1
        if inp.ndim != 3 or inp.shape[1] != expected_rows:
            raise ValueError(
                f"Expected input shape (batch, {expected_rows}, seq), got {tuple(inp.shape)}"
            )

    def _embed(self, inp: mx.array) -> mx.array:
        semantic_ids = inp[:, 0]
        codebook_rows = inp[:, 1:]
        vq_embeds = []
        for i in range(self.num_codebooks):
            offset_ids = (
                codebook_rows[:, i] + i * self.config.audio_decoder_config.vocab_size
            )
            vq_embeds.append(self.codebook_embeddings(offset_ids))
        vq_sum = mx.stack(vq_embeds, axis=0).sum(axis=0)
        semantic_mask = (semantic_ids >= self.config.semantic_start_token_id) & (
            semantic_ids <= self.config.semantic_end_token_id
        )
        semantic_embeds = self.embeddings(semantic_ids)
        combined = semantic_embeds + mx.where(
            semantic_mask[:, :, None],
            vq_sum,
            mx.zeros_like(vq_sum),
        )
        scale = math.sqrt(self.num_codebooks + 1)
        return mx.where(semantic_mask[:, :, None], combined / scale, semantic_embeds)

    def __call__(
        self, inp: mx.array, *, cache: KVCache | None = None
    ) -> ForwardResult:
        self._validate_input_shape(inp)
        x = self._embed(inp)
        for i, layer in enumerate(self.layers):
            x = layer(x, cache=cache, layer_idx=i)
        slow_out = self.norm(x)
        logits = self.embeddings.as_linear(slow_out)
        return ForwardResult(
            logits=logits, hidden_states=self.fast_project_in(slow_out)
        )

    def fast_forward(
        self, hidden_state: mx.array, previous_codebooks: mx.array
    ) -> mx.array:
        hidden_state = (
            hidden_state[:, -1:, :]
            if hidden_state.ndim == 3
            else hidden_state[:, None, :]
        )
        if previous_codebooks.size > 0:
            x = mx.concatenate(
                [hidden_state, self.fast_embeddings(previous_codebooks)], axis=1
            )
        else:
            x = hidden_state
        for layer in self.fast_layers:
            x = layer(x)
        return self.fast_output(self.fast_norm(x)[:, -1])
