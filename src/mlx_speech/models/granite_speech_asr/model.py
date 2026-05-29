"""Full Granite Speech ASR model assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .checkpoint import (
    AlignmentReport,
    load_checkpoint_into_model,
    load_granite_speech_checkpoint,
)
from .config import GraniteSpeechConfig
from .encoder import GraniteSpeechEncoder
from .feature_extraction import GraniteSpeechFeatureExtractor
from .language_model import GraniteCausalLM, GraniteKVCache
from .projector import GraniteSpeechProjector
from .tokenizer import GraniteSpeechTokenizer


@dataclass(frozen=True)
class GraniteSpeechModelBundle:
    """Loaded Granite Speech runtime components without retaining raw weights."""

    model: "GraniteSpeechModel"
    feature_extractor: GraniteSpeechFeatureExtractor
    tokenizer: GraniteSpeechTokenizer
    config: GraniteSpeechConfig
    alignment: AlignmentReport
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    transposed_keys: tuple[str, ...]


class GraniteSpeechModel(nn.Module):
    """Encoder, projector, and local Granite LM with checkpoint-shaped paths."""

    def __init__(self, config: GraniteSpeechConfig):
        super().__init__()
        self.config = config
        self.encoder = GraniteSpeechEncoder(config.encoder)
        self.projector = GraniteSpeechProjector(config)
        self.language_model = GraniteCausalLM(
            config.text,
            tie_word_embeddings=config.tie_word_embeddings,
        )
        self.audio_token_id = config.audio_token_index

    @property
    def layers(self):
        return self.language_model.layers

    def get_audio_features(self, input_features: mx.array) -> mx.array:
        return self.projector(self.encoder(input_features))

    def embed_input_ids(self, input_ids: mx.array) -> mx.array:
        return self.language_model.model.embed_tokens(input_ids)

    def __call__(
        self,
        input_ids: mx.array | None = None,
        *,
        inputs_embeds: mx.array | None = None,
        input_embeddings: mx.array | None = None,
        kv_cache: GraniteKVCache | None = None,
    ) -> mx.array:
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            input_embeddings=input_embeddings,
            kv_cache=kv_cache,
        )

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        strict: bool = True,
    ) -> GraniteSpeechModelBundle:
        return load_granite_speech_model(model_dir, dtype=dtype, strict=strict)

    @classmethod
    def from_path(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
        strict: bool = True,
    ) -> GraniteSpeechModelBundle:
        return cls.from_dir(model_dir, dtype=dtype, strict=strict)


def load_granite_speech_model(
    model_dir: str | Path,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    strict: bool = True,
) -> GraniteSpeechModelBundle:
    model_dir = Path(model_dir)
    checkpoint = load_granite_speech_checkpoint(model_dir)
    model = GraniteSpeechModel(checkpoint.config)
    alignment = load_checkpoint_into_model(model, checkpoint, strict=strict)
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())

    return GraniteSpeechModelBundle(
        model=model,
        feature_extractor=GraniteSpeechFeatureExtractor.from_dir(model_dir),
        tokenizer=GraniteSpeechTokenizer.from_dir(model_dir),
        config=checkpoint.config,
        alignment=alignment,
        source_files=checkpoint.source_files,
        skipped_keys=checkpoint.skipped_keys,
        transposed_keys=checkpoint.transposed_keys,
    )
