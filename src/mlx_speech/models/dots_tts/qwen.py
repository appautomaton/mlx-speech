"""dots.tts Qwen2.5 contextual trunk and EOS projection."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .._qwen2 import Qwen2KVCache, Qwen2Model
from .config import DotsTTSQwenConfig


class DotsTTSEOSHead(nn.Module):
    """Two-layer SiLU head producing continue/EOS logits."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, 2, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(hidden_states)))


@dataclass(frozen=True)
class DotsTTSQwenOutput:
    last_hidden_state: mx.array
    eos_logits: mx.array
    cache: Qwen2KVCache
    logits: mx.array | None = None

    @property
    def past_key_values(self) -> Qwen2KVCache:
        return self.cache


class DotsTTSQwen(nn.Module):
    """Qwen2.5 trunk with tied token logits and the dots.tts EOS head."""

    def __init__(self, config: DotsTTSQwenConfig):
        super().__init__()
        if not config.tie_word_embeddings:
            raise ValueError("dots.tts Qwen requires tied token embeddings.")
        self.config = config
        self.model = Qwen2Model(config, rotary_dtype_policy="query")
        self.eos_proj = DotsTTSEOSHead(config.hidden_size)

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Embedding:
        """Return the same embedding module used by the output projection."""

        return self.model.embed_tokens

    def project_logits(self, hidden_states: mx.array) -> mx.array:
        return self.model.tied_logits(hidden_states)

    def eos_logits(self, hidden_states: mx.array) -> mx.array:
        return self.eos_proj(hidden_states)

    def eos_probabilities(self, hidden_states: mx.array) -> mx.array:
        return mx.softmax(self.eos_logits(hidden_states), axis=-1)[..., 1]

    def should_stop(
        self,
        hidden_states: mx.array,
        *,
        threshold: float = 0.8,
    ) -> mx.array:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"EOS threshold must be in [0, 1], got {threshold}.")
        return self.eos_probabilities(hidden_states) > threshold

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: Qwen2KVCache | None = None,
        cache_capacity: int | None = None,
        request_logits: bool = True,
    ) -> DotsTTSQwenOutput:
        output = self.model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            cache_capacity=cache_capacity,
        )
        hidden_states = output.last_hidden_state
        logits = self.project_logits(hidden_states) if request_logits else None
        return DotsTTSQwenOutput(
            last_hidden_state=hidden_states,
            eos_logits=self.eos_logits(hidden_states),
            cache=output.cache,
            logits=logits,
        )

    def step(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        cache: Qwen2KVCache | None = None,
        cache_capacity: int | None = None,
        request_logits: bool = False,
    ) -> DotsTTSQwenOutput:
        """Run a cache-aware contextual step without vocabulary logits by default."""

        return self(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            cache=cache,
            cache_capacity=cache_capacity,
            request_logits=request_logits,
        )


DotsTTSQwenModel = DotsTTSQwen

__all__ = [
    "DotsTTSEOSHead",
    "DotsTTSQwen",
    "DotsTTSQwenModel",
    "DotsTTSQwenOutput",
]
