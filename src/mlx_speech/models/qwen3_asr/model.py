"""Qwen3-ASR multimodal model assembly."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .audio_encoder import Qwen3ASRAudioEncoder
from .config import Qwen3ASRConfig
from .text_decoder import (
    Qwen3ASRDecodeCompute,
    Qwen3ASRTextCausalLMOutput,
    Qwen3ASRTextForCausalLM,
    Qwen3ASRTextKVCache,
)


@dataclass(frozen=True)
class Qwen3ASRModelOutput:
    logits: mx.array
    past_key_values: Qwen3ASRTextKVCache | None = None


class Qwen3ASRModel(nn.Module):
    """Audio tower plus Qwen3 text decoder.

    Serves ASR generation and, with a time-class head, Qwen3-ForcedAligner.
    """

    def __init__(self, config: Qwen3ASRConfig):
        super().__init__()
        if config.audio_config.output_dim != config.text_config.hidden_size:
            raise ValueError(
                "Qwen3-ASR audio output_dim must match text hidden_size: "
                f"{config.audio_config.output_dim} vs {config.text_config.hidden_size}."
            )
        self.config = config
        self.audio_tower = Qwen3ASRAudioEncoder(config.audio_config)
        self.text_decoder = Qwen3ASRTextForCausalLM(
            config.text_config,
            lm_head_size=config.thinker_config.lm_head_size,
        )
        self.audio_token_id = config.audio_token_id

    def get_audio_features(
        self,
        input_features: mx.array,
        *,
        feature_attention_mask: mx.array | None = None,
    ) -> mx.array:
        features = mx.array(input_features)
        if features.ndim != 3:
            raise ValueError(f"Expected Qwen3-ASR input_features [B, mel, frames], got {features.shape}.")
        feature_lengths = _feature_lengths(features, feature_attention_mask)
        outputs = []
        for index, feature_len in enumerate(feature_lengths):
            encoded = self.audio_tower(
                features[index],
                feature_lens=int(feature_len),
            ).last_hidden_state
            outputs.append(encoded)
        return mx.concatenate(outputs, axis=0) if len(outputs) > 1 else outputs[0]

    def get_audio_features_batch(self, input_features: list[mx.array]) -> list[mx.array]:
        """Return per-clip audio tokens from unpadded [mel, frames] features."""
        return self.audio_tower.encode_batch(input_features)

    def embed_input_ids(
        self,
        input_ids: mx.array,
        *,
        replacement_id: int = 0,
    ) -> mx.array:
        masked_ids = mask_audio_token_ids(
            input_ids,
            audio_token_id=self.audio_token_id,
            replacement_id=replacement_id,
        )
        return self.text_decoder.model.embed_tokens(masked_ids)

    def prepare_inputs_embeds(
        self,
        input_ids: mx.array,
        audio_features: mx.array | list[mx.array],
        *,
        replacement_id: int = 0,
    ) -> mx.array:
        token_embeddings = self.embed_input_ids(input_ids, replacement_id=replacement_id)
        return replace_audio_embeddings(
            input_ids,
            token_embeddings,
            audio_features,
            audio_token_id=self.audio_token_id,
        )

    def prefill(
        self,
        *,
        inputs_embeds: mx.array,
        max_cache_len: int,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        compute: Qwen3ASRDecodeCompute | None = None,
        last_logits_only: bool = False,
    ) -> Qwen3ASRTextCausalLMOutput:
        kv_cache = Qwen3ASRTextKVCache.allocate(
            self.config.text_config,
            batch_size=int(inputs_embeds.shape[0]),
            max_length=max_cache_len,
            dtype=inputs_embeds.dtype,
        )
        return self.text_decoder.prefill(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
            compute=compute,
            last_logits_only=last_logits_only,
        )

    def decode_step(
        self,
        *,
        input_ids: mx.array,
        kv_cache: Qwen3ASRTextKVCache,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        compute: Qwen3ASRDecodeCompute | None = None,
    ) -> Qwen3ASRTextCausalLMOutput:
        return self.text_decoder.decode_step(
            input_ids=input_ids,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            compute=compute,
        )

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        inputs_embeds: mx.array | None = None,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
    ) -> Qwen3ASRTextCausalLMOutput:
        return self.text_decoder(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )


def mask_audio_token_ids(
    input_ids: mx.array,
    *,
    audio_token_id: int,
    replacement_id: int = 0,
) -> mx.array:
    return mx.where(
        input_ids == int(audio_token_id),
        mx.array(int(replacement_id), dtype=input_ids.dtype),
        input_ids,
    )


def replace_audio_embeddings(
    input_ids: mx.array,
    token_embeddings: mx.array,
    audio_features: mx.array | list[mx.array],
    *,
    audio_token_id: int,
    audio_token_positions: list[list[int]] | None = None,
) -> mx.array:
    """Splice audio tokens; host-known positions avoid reading back input_ids."""
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids [B, T], got {input_ids.shape}.")
    if token_embeddings.ndim != 3:
        raise ValueError(f"Expected token_embeddings [B, T, C], got {token_embeddings.shape}.")
    batch_size = int(input_ids.shape[0])
    if token_embeddings.shape[:2] != input_ids.shape:
        raise ValueError("token_embeddings batch/sequence shape must match input_ids.")
    if isinstance(audio_features, list):
        if len(audio_features) != batch_size:
            raise ValueError("Expected one audio_features array per prompt row.")
        rows = audio_features
    elif batch_size == 1:
        rows = [audio_features]
    else:
        raise ValueError("Batched prompts require a list of per-row audio_features arrays.")

    if audio_token_positions is not None and len(audio_token_positions) != batch_size:
        raise ValueError("Expected one audio_token_positions list per prompt row.")
    output = mx.array(token_embeddings)
    for row, row_features in enumerate(rows):
        features = mx.array(row_features)
        if features.ndim == 3:
            if int(features.shape[0]) != 1:
                raise ValueError("Qwen3-ASR v0 supports one audio feature batch at a time.")
            features = features[0]
        if features.ndim != 2:
            raise ValueError(f"Expected audio_features [audio_tokens, hidden], got {features.shape}.")
        if int(features.shape[-1]) != int(token_embeddings.shape[-1]):
            raise ValueError(
                f"Audio feature width {features.shape[-1]} does not match text hidden width "
                f"{token_embeddings.shape[-1]}."
            )
        positions = (
            audio_token_positions[row] if audio_token_positions is not None
            else _audio_token_positions(input_ids, audio_token_id=audio_token_id, row=row)
        )
        if len(positions) != int(features.shape[0]):
            raise ValueError(
                "Qwen3-ASR audio placeholder count mismatch: "
                f"input_ids has {len(positions)}, audio_features has {features.shape[0]}."
            )
        output[row, mx.array(positions, dtype=mx.int32), :] = features
    return output


def count_audio_tokens(input_ids: mx.array, *, audio_token_id: int) -> int:
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids [B, T], got {input_ids.shape}.")
    return sum(
        len(_audio_token_positions(input_ids, audio_token_id=audio_token_id, row=row))
        for row in range(int(input_ids.shape[0]))
    )


def _audio_token_positions(
    input_ids: mx.array, *, audio_token_id: int, row: int = 0
) -> list[int]:
    if input_ids.ndim != 2:
        raise ValueError(f"Expected input_ids [B, T], got {input_ids.shape}.")
    mask = input_ids[row] == int(audio_token_id)
    return [int(index) for index, value in enumerate(mask.tolist()) if bool(value)]


def _feature_lengths(
    input_features: mx.array,
    feature_attention_mask: mx.array | None,
) -> list[int]:
    if feature_attention_mask is None:
        return [int(input_features.shape[-1])] * int(input_features.shape[0])
    if feature_attention_mask.ndim != 2:
        raise ValueError(
            f"Expected feature_attention_mask [B, frames], got {feature_attention_mask.shape}."
        )
    if int(feature_attention_mask.shape[0]) != int(input_features.shape[0]):
        raise ValueError(
            "feature_attention_mask batch size does not match input_features: "
            f"{feature_attention_mask.shape[0]} vs {input_features.shape[0]}."
        )
    return [int(value) for value in feature_attention_mask.sum(axis=-1).tolist()]
