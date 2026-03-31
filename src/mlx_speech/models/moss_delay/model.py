"""MLX model modules for MossTTSDelay."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..moss_common import GlobalKVCache
from ..moss_common.model import Qwen3Model, Qwen3ModelOutput
from .config import MossTTSDelayConfig


@dataclass(frozen=True)
class MossTTSDelayOutput:
    """Return type for the MLX-side MossTTSDelay model."""

    last_hidden_state: mx.array
    logits_all: tuple[mx.array, ...]
    hidden_states: tuple[mx.array, ...] | None = None


class MossTTSDelayModel(nn.Module):
    """Single-backbone delay-pattern model for TTSD-family checkpoints."""

    def __init__(self, config: MossTTSDelayConfig):
        super().__init__()
        self.config = config
        self.channels = config.channels
        self.activation_dtype = (
            mx.bfloat16 if config.extra.get("quantization") is not None else mx.float32
        )
        self.language_model = Qwen3Model(config.language_config)
        self.emb_ext = [
            nn.Embedding(config.audio_embedding_vocab_size, config.hidden_size)
            for _ in range(config.n_vq)
        ]
        self.lm_heads = [nn.Linear(config.hidden_size, config.vocab_size, bias=False)]
        for _ in range(config.n_vq):
            self.lm_heads.append(
                nn.Linear(config.hidden_size, config.audio_embedding_vocab_size, bias=False)
            )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.embed_tokens

    def _compute_input_embeddings(self, input_ids: mx.array) -> mx.array:
        if input_ids.ndim != 3 or int(input_ids.shape[-1]) != self.channels:
            raise ValueError(
                "Expected `input_ids` with shape (batch, seq, 1 + n_vq), "
                f"got {input_ids.shape}."
            )

        inputs_embeds = self.get_input_embeddings()(input_ids[..., 0]).astype(
            self.activation_dtype
        )
        for layer_index, embed_layer in enumerate(self.emb_ext):
            inputs_embeds = (
                inputs_embeds + embed_layer(input_ids[..., layer_index + 1])
            ).astype(self.activation_dtype)
        return inputs_embeds

    def _compute_logits(self, hidden_states: mx.array) -> tuple[mx.array, ...]:
        logits_all: list[mx.array] = []
        for layer_index, head in enumerate(self.lm_heads):
            logits = head(hidden_states).astype(self.activation_dtype)
            if layer_index > 0:
                logits[..., -1] = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
            logits_all.append(logits)
        return tuple(logits_all)

    def _build_output(
        self,
        backbone: Qwen3ModelOutput,
    ) -> MossTTSDelayOutput:
        return MossTTSDelayOutput(
            last_hidden_state=backbone.last_hidden_state,
            logits_all=self._compute_logits(backbone.last_hidden_state),
            hidden_states=backbone.hidden_states,
        )

    def forward_backbone(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")
        if input_ids is not None:
            inputs_embeds = self._compute_input_embeddings(input_ids)
        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        output_hidden_states: bool = False,
    ) -> MossTTSDelayOutput:
        backbone = self.forward_backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )
        return self._build_output(backbone)

    def prefill(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        max_cache_len: int,
        output_hidden_states: bool = False,
    ) -> tuple[MossTTSDelayOutput, GlobalKVCache]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if input_ids is not None:
            inputs_embeds = self._compute_input_embeddings(input_ids)

        kv_cache = GlobalKVCache.allocate(
            self.language_model.config,
            batch_size=int(inputs_embeds.shape[0]),
            max_length=max_cache_len,
            dtype=mx.float32,
        )
        backbone = self.language_model.prefill(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
        )
        kv_cache.prompt_length = int(inputs_embeds.shape[1])
        return self._build_output(backbone), kv_cache

    def decode_step(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        kv_cache: GlobalKVCache,
        output_hidden_states: bool = False,
    ) -> MossTTSDelayOutput:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of `input_ids` or `inputs_embeds`.")

        if input_ids is not None:
            inputs_embeds = self._compute_input_embeddings(input_ids)

        backbone = self.language_model.decode_step(
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
            output_hidden_states=output_hidden_states,
        )
        return self._build_output(backbone)
