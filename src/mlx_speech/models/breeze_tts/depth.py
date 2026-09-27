"""Depth decoder: one frame of residual codebooks, then a fresh cache."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_speech.models._cache import BoundedKVCache

from .backbone import EmbeddingTable
from .causal import (
    DecoderLayer,
    RMSNorm,
    RotaryEmbedding,
    make_decoder_cache,
    run_causal_decoder,
)
from .config import BreezeRuntimeConfig


class CodebookHeads(nn.Module):
    """Position-specific heads. Head 0 predicts codebook 1."""

    def __init__(self, hidden_size: int, num_codebooks: int, vocab_size: int) -> None:
        super().__init__()
        self.weight = mx.zeros((num_codebooks - 1, hidden_size, vocab_size))

    def __call__(self, hidden_states: mx.array, cache_position: mx.array) -> mx.array:
        if cache_position.ndim != 1 or int(cache_position.shape[0]) != int(
            hidden_states.shape[1]
        ):
            raise ValueError(
                "Codebook positions must match the hidden sequence, got "
                f"{tuple(cache_position.shape)} vs {tuple(hidden_states.shape)}."
            )
        selected = self.weight[cache_position - 1]
        return mx.matmul(hidden_states[:, :, None, :], selected[None, :, :, :]).squeeze(
            axis=2
        )


class _DepthModel(nn.Module):
    def __init__(self, config: BreezeRuntimeConfig) -> None:
        super().__init__()
        self.embed_tokens = EmbeddingTable(
            config.audio_embedding_rows, config.audio_embed_size
        )
        self.inputs_embeds_projector = nn.Linear(
            config.audio_embed_size, config.depth.hidden_size, bias=False
        )
        self.layers = [
            DecoderLayer(config.depth) for _ in range(config.depth.num_hidden_layers)
        ]
        self.norm = RMSNorm(config.depth.hidden_size, config.depth.rms_norm_eps)
        self.rope = RotaryEmbedding.from_config(config.depth)

    def embed_code(
        self, code_ids: mx.array, codebook_index: int, vocab_size: int
    ) -> mx.array:
        return self.embed_tokens(code_ids + codebook_index * vocab_size)

    def forward(self, inputs_embeds: mx.array, cache: list[BoundedKVCache]) -> mx.array:
        projected = self.inputs_embeds_projector(inputs_embeds)
        return run_causal_decoder(self.layers, self.norm, self.rope, projected, cache)


class BreezeDepthDecoder(nn.Module):
    def __init__(self, config: BreezeRuntimeConfig) -> None:
        super().__init__()
        self.config = config
        self.model = _DepthModel(config)
        self.codebooks_head = CodebookHeads(
            config.depth.hidden_size, config.num_codebooks, config.vocab_size
        )

    def make_cache(self, *, batch_size: int, dtype: mx.Dtype) -> list[BoundedKVCache]:
        return make_decoder_cache(self.config.depth, batch_size=batch_size, dtype=dtype)

    def reset_cache(self, cache: list[BoundedKVCache]) -> None:
        for layer_cache in cache:
            layer_cache.restore_offset(0)

    def embed_code(self, code_ids: mx.array, codebook_index: int) -> mx.array:
        if codebook_index < 0 or codebook_index >= self.config.num_codebooks:
            raise ValueError(f"Codebook index {codebook_index} is outside the frame.")
        return self.model.embed_code(code_ids, codebook_index, self.config.vocab_size)

    def begin_frame(
        self,
        backbone_hidden: mx.array,
        first_code_embedding: mx.array,
        cache: list[BoundedKVCache],
    ) -> mx.array:
        if cache[0].offset != 0:
            raise ValueError("A depth frame must start from an empty cache.")
        inputs = mx.stack([backbone_hidden, first_code_embedding], axis=1)
        hidden = self.model.forward(inputs, cache)
        positions = mx.array([1], dtype=mx.int32)
        return self.codebooks_head(hidden[:, 1:2, :], positions)[:, 0, :]

    def step(
        self,
        code_embedding: mx.array,
        *,
        codebook_index: int,
        cache: list[BoundedKVCache],
    ) -> mx.array:
        position = codebook_index + 1
        hidden = self.model.forward(code_embedding[:, None, :], cache)
        positions = mx.array([position], dtype=mx.int32)
        return self.codebooks_head(hidden[:, -1:, :], positions)[:, 0, :]


def expected_depth_checkpoint_weights(
    config: BreezeRuntimeConfig,
) -> dict[str, tuple[int, ...]]:
    decoder = config.depth
    hidden = decoder.hidden_size
    intermediate = decoder.intermediate_size
    query = decoder.num_attention_heads * decoder.head_dim
    key = decoder.num_key_value_heads * decoder.head_dim
    weights: dict[str, tuple[int, ...]] = {
        "depth_decoder.model.embed_tokens.weight": (
            config.audio_embedding_rows,
            config.audio_embed_size,
        ),
        "depth_decoder.model.inputs_embeds_projector.weight": (
            hidden,
            config.audio_embed_size,
        ),
        "depth_decoder.model.norm.weight": (hidden,),
        "depth_decoder.codebooks_head.weight": (
            config.num_codebooks - 1,
            hidden,
            config.vocab_size,
        ),
    }
    for index in range(decoder.num_hidden_layers):
        prefix = f"depth_decoder.model.layers.{index}"
        weights.update(
            {
                f"{prefix}.self_attn.q_proj.weight": (query, hidden),
                f"{prefix}.self_attn.k_proj.weight": (key, hidden),
                f"{prefix}.self_attn.v_proj.weight": (key, hidden),
                f"{prefix}.self_attn.o_proj.weight": (hidden, query),
                f"{prefix}.input_layernorm.weight": (hidden,),
                f"{prefix}.post_attention_layernorm.weight": (hidden,),
                f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
            }
        )
    return weights
