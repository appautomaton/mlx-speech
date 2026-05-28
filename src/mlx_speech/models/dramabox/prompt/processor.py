"""DramaBox `EmbeddingsProcessor` — pipeline orchestrator.

Glues `FeatureExtractorV2` and `Embeddings1DConnector` into the single
`process_hidden_states` entrypoint used by the upstream TTS pipeline. The
output `audio_encoding` is the `a_ctx` tensor that feeds the DiT's cross-
attention layers.

Reference: `.references/DramaBox/ltx2/ltx_core/text_encoders/gemma/embeddings_processor.py:30-89`
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .additive_mask import convert_to_additive_mask
from .connector import Embeddings1DConnector
from .feature_extractor import FeatureExtractorV2


@dataclass
class EmbeddingsProcessorOutput:
    """`a_ctx` and the post-connector binary mask.

    The DramaBox audio path uses only `audio_encoding`; `attention_mask` is
    returned for completeness (always all-ones after the connector replaces
    padded slots with registers).
    """

    audio_encoding: mx.array
    attention_mask: mx.array


class EmbeddingsProcessor(nn.Module):
    """`feature_extractor` + `audio_connector` pipeline.

    For DramaBox there is no video branch — the upstream
    `video_aggregate_embed` is absent from the checkpoint and the audio path
    is the only consumer. We expose `process_hidden_states` directly; the
    `create_embeddings` path used by the trainer is out of scope (inference
    only).
    """

    def __init__(
        self,
        *,
        feature_extractor: FeatureExtractorV2,
        audio_connector: Embeddings1DConnector,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.audio_connector = audio_connector

    def __call__(
        self,
        hidden_states: list[mx.array] | tuple[mx.array, ...],
        attention_mask: mx.array,
    ) -> EmbeddingsProcessorOutput:
        """Run the full pipeline.

        Args:
            hidden_states: list of 49 Gemma hidden states each ``[B, T, D]``.
            attention_mask: ``[B, T]`` binary mask (1=valid, 0=pad).

        Returns:
            `EmbeddingsProcessorOutput(audio_encoding, attention_mask)`.
            `audio_encoding` has shape ``[B, T, 2048]``.
        """
        audio_feats = self.feature_extractor(hidden_states, attention_mask)
        additive_mask = convert_to_additive_mask(attention_mask, audio_feats.dtype)
        audio_encoded, post_mask = self.audio_connector(audio_feats, additive_mask)

        # Post-connector binary mask: the connector returns an all-zero
        # additive mask (every slot is valid). Convert to binary [B, T] form
        # so downstream consumers don't depend on the additive convention.
        B, _, _, T = post_mask.shape
        binary_post = mx.ones((B, T), dtype=mx.int32)
        return EmbeddingsProcessorOutput(
            audio_encoding=audio_encoded,
            attention_mask=binary_post,
        )


__all__ = ["EmbeddingsProcessor", "EmbeddingsProcessorOutput"]
