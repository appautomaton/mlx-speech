"""Qwen3 backbone used to predict the first codebook, including EOS."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_speech.models._cache import BoundedKVCache

from .causal import (
    DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    make_decoder_cache,
    run_causal_decoder,
)
from .config import BreezeRuntimeConfig


class EmbeddingTable(nn.Module):
    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.weight = mx.zeros((rows, cols))

    def __call__(self, token_ids: mx.array) -> mx.array:
        return self.weight[token_ids]


class _AudioEmbeddings(nn.Module):
    def __init__(self, rows: int, cols: int) -> None:
        super().__init__()
        self.embed_audio_tokens = EmbeddingTable(rows, cols)


class BreezeBackbone(nn.Module):
    def __init__(self, config: BreezeRuntimeConfig) -> None:
        super().__init__()
        if config.audio_embed_size != config.backbone.hidden_size:
            raise ValueError("Backbone audio projector is not implemented.")
        self.config = config
        self.embed_tokens = _AudioEmbeddings(
            config.audio_embedding_rows, config.audio_embed_size
        )
        self.layers = [
            DecoderLayer(config.backbone)
            for _ in range(config.backbone.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.backbone.hidden_size, config.backbone.rms_norm_eps)
        self.rope = RotaryEmbedding.from_config(config.backbone)

    def make_cache(self, *, batch_size: int, dtype: mx.Dtype) -> list[BoundedKVCache]:
        return make_decoder_cache(
            self.config.backbone, batch_size=batch_size, dtype=dtype
        )

    def embed_audio(self, code_ids: mx.array) -> mx.array:
        """Sum one embedding per codebook. ``code_ids`` is (batch, sequence, codebooks)."""

        if code_ids.ndim != 3 or int(code_ids.shape[-1]) != self.config.num_codebooks:
            raise ValueError(
                "Audio code ids must have shape (batch, sequence, "
                f"{self.config.num_codebooks}), got {tuple(code_ids.shape)}."
            )
        offsets = mx.arange(self.config.num_codebooks) * self.config.vocab_size
        return mx.sum(self.embed_tokens.embed_audio_tokens(code_ids + offsets), axis=2)

    def __call__(
        self,
        inputs_embeds: mx.array,
        cache: list[BoundedKVCache],
        *,
        position_ids: mx.array | None = None,
    ) -> mx.array:
        return run_causal_decoder(
            self.layers,
            self.norm,
            self.rope,
            inputs_embeds,
            cache,
            position_ids=position_ids,
        )


def expected_backbone_checkpoint_weights(
    config: BreezeRuntimeConfig,
) -> dict[str, tuple[int, ...]]:
    decoder = config.backbone
    hidden = decoder.hidden_size
    intermediate = decoder.intermediate_size
    query = decoder.num_attention_heads * decoder.head_dim
    key = decoder.num_key_value_heads * decoder.head_dim
    weights: dict[str, tuple[int, ...]] = {
        "backbone_model.embed_tokens.embed_audio_tokens.weight": (
            config.audio_embedding_rows,
            config.audio_embed_size,
        ),
        "backbone_model.norm.weight": (hidden,),
        "lm_head.weight": (config.vocab_size + 1, hidden),
        "embed_text_tokens.weight": (config.text_vocab_size, hidden),
    }
    for index in range(decoder.num_hidden_layers):
        prefix = f"backbone_model.layers.{index}"
        weights.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (query, hidden),
                f"{prefix}.self_attn.k_proj.weight": (key, hidden),
                f"{prefix}.self_attn.v_proj.weight": (key, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, query),
                f"{prefix}.self_attn.q_norm.weight": (decoder.head_dim,),
                f"{prefix}.self_attn.k_norm.weight": (decoder.head_dim,),
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return weights
