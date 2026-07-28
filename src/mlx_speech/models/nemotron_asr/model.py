"""Assembled pure-MLX Nemotron 3.5 ASR model and offline inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .checkpoint import load_nemotron_checkpoint, load_state_dict_strict
from .config import NemotronASRConfig
from .encoder import FastConformerEncoder
from .feature_extraction import NemotronPreprocessor
from .prompt import apply_language_prompt, build_prompt_kernel
from .tokenizer import NemotronTokenizer
from .transducer import JointNetwork, PredictionNetwork, greedy_transducer_decode


@dataclass(frozen=True)
class NemotronASRResult:
    text: str
    tokens: tuple[int, ...]
    language: str
    detected_language: str | None
    frame_indices: tuple[int, ...]


class NemotronASRModel(nn.Module):
    """Checkpoint-compatible model tree with batch-one greedy inference."""

    def __init__(self, config: NemotronASRConfig) -> None:
        super().__init__()
        self.config = config
        self.preprocessor = NemotronPreprocessor(config.preprocessor)
        self.encoder = FastConformerEncoder(config.encoder)
        self.prompt_kernel = build_prompt_kernel(config.encoder.d_model, config.prompt)
        self.decoder = PredictionNetwork(config.decoder)
        self.joint = JointNetwork(config.joint)
        self.tokenizer = NemotronTokenizer(config.vocabulary)
        self.blank_id = config.decoder.vocab_size

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "NemotronASRModel":
        checkpoint = load_nemotron_checkpoint(model_dir)
        model = cls(checkpoint.config)
        load_state_dict_strict(model, checkpoint.state_dict)
        model.eval()
        mx.eval(model.parameters())
        return model

    def encode(
        self,
        features: mx.array,
        lengths: mx.array,
        *,
        language: str,
        att_context_size: tuple[int, int] | None = None,
    ) -> tuple[mx.array, mx.array]:
        dtype = self.encoder.pre_encode.out.weight.dtype
        encoded, encoded_lengths = self.encoder(
            features.astype(dtype),
            lengths,
            att_context_size or self.config.default_att_context_size,
        )
        encoded = apply_language_prompt(
            encoded,
            language,
            self.config.prompt,
            self.prompt_kernel,
        )
        return encoded, encoded_lengths

    def transcribe_features(
        self,
        features: mx.array,
        lengths: mx.array,
        *,
        language: str | None = None,
        att_context_size: tuple[int, int] | None = None,
        strip_language_tags: bool = True,
    ) -> NemotronASRResult:
        selected_language = language or self.config.default_language
        encoded, encoded_lengths = self.encode(
            features,
            lengths,
            language=selected_language,
            att_context_size=att_context_size,
        )
        valid_length = int(encoded_lengths[0].item())
        decoded = greedy_transducer_decode(
            encoded[:, :valid_length],
            self.decoder,
            self.joint,
            blank_id=self.blank_id,
            max_symbols=self.config.max_symbols,
        )
        detected = self.tokenizer.detected_language(decoded.tokens)
        return NemotronASRResult(
            text=self.tokenizer.decode(
                decoded.tokens, strip_language_tags=strip_language_tags
            ),
            tokens=decoded.tokens,
            language=selected_language,
            detected_language=detected,
            frame_indices=decoded.frame_indices,
        )

    def transcribe(
        self,
        audio: mx.array | np.ndarray,
        *,
        sample_rate: int = 16_000,
        language: str | None = None,
        att_context_size: tuple[int, int] | None = None,
        strip_language_tags: bool = True,
    ) -> NemotronASRResult:
        if sample_rate != self.config.preprocessor.sample_rate:
            raise ValueError(
                f"Nemotron requires {self.config.preprocessor.sample_rate} Hz audio; "
                f"got {sample_rate} Hz"
            )
        features, lengths = self.preprocessor(audio)
        return self.transcribe_features(
            features,
            lengths,
            language=language,
            att_context_size=att_context_size,
            strip_language_tags=strip_language_tags,
        )


__all__ = ["NemotronASRModel", "NemotronASRResult"]
