"""Assembled pure-MLX Audio8-ASR-Infinite model."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .audio_encoder import Audio8Encoder
from .checkpoint import load_audio8_checkpoint, load_checkpoint_into_model
from .config import Audio8Config
from .feature_extraction import Audio8FeatureConfig, Audio8FeatureExtractor
from .text_decoder import Audio8LanguageModel, Audio8TimeEmbedding
from .tokenizer import Audio8Tokenizer


class Audio8Projector(nn.Module):
    def __init__(self, config: Audio8Config) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(config.projection_size, config.text_config.hidden_size, bias=False)
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=False)

    def __call__(self, audio_features: mx.array) -> mx.array:
        return self.linear_2(nn.gelu(self.linear_1(audio_features)))


class Audio8ASRModel(nn.Module):
    def __init__(self, config: Audio8Config) -> None:
        super().__init__()
        self.config = config
        self.audio_tower = Audio8Encoder(config.audio_config)
        self.language_model = Audio8LanguageModel(config.text_config)
        self.multi_modal_projector = Audio8Projector(config)
        self.time_embedding = Audio8TimeEmbedding(config.text_config.hidden_size)
        self.frame_len_embedding = (
            nn.Embedding(len(config.supported_frame_lens), config.text_config.hidden_size)
            if config.use_frame_len_embedding
            else None
        )
        self.semantic_vad_heads = [
            nn.Linear(config.text_config.hidden_size, config.semantic_vad_num_classes)
            for _ in config.semantic_vad_horizons_seconds
        ]
        self.tokenizer: Audio8Tokenizer | None = None
        self.feature_extractor = Audio8FeatureExtractor(
            Audio8FeatureConfig(feature_size=config.audio_config.num_mel_bins)
        )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Audio8ASRModel":
        checkpoint = load_audio8_checkpoint(model_dir)
        model = cls(checkpoint.config)
        load_checkpoint_into_model(model, checkpoint, strict=True)
        model.tokenizer = Audio8Tokenizer.from_dir(model_dir)
        model.feature_extractor = Audio8FeatureExtractor(
            Audio8FeatureConfig.from_dir(model_dir)
        )
        model.eval()
        mx.eval(model.parameters())
        return model

    def build_t_cond(
        self, *, frame_len: int, num_delay_tokens: int, dtype: mx.Dtype
    ) -> mx.array:
        t_cond = self.time_embedding(num_delay_tokens, dtype=dtype)
        if self.frame_len_embedding is not None:
            index = self.config.supported_frame_lens.index(frame_len)
            t_cond = t_cond + self.frame_len_embedding(
                mx.array([index], dtype=mx.int32)
            )[None, :, :]
        return t_cond

    def group_audio_hidden_states(
        self,
        hidden_states: mx.array,
        *,
        frame_len: int,
        target_token_count: int | None = None,
    ) -> mx.array:
        self.config.clock_ms(frame_len)
        batch, frames, hidden = hidden_states.shape
        if frames % frame_len:
            raise ValueError("Audio8 streaming audio frames must form complete clock groups")
        tokens = frames // frame_len
        grouped = hidden_states.reshape(batch, tokens, frame_len, hidden)
        if frame_len < self.config.max_frame_len:
            grouped = mx.pad(
                grouped,
                ((0, 0), (0, 0), (0, self.config.max_frame_len - frame_len), (0, 0)),
            )
        grouped = grouped.reshape(batch, tokens, self.config.projection_size)
        if target_token_count is not None:
            if tokens >= target_token_count:
                return grouped[:, :target_token_count]
            grouped = mx.pad(
                grouped, ((0, 0), (0, target_token_count - tokens), (0, 0))
            )
        return grouped

    def project_audio(self, hidden_states: mx.array, *, frame_len: int) -> mx.array:
        return self.multi_modal_projector(
            self.group_audio_hidden_states(hidden_states, frame_len=frame_len)
        )

    def build_text_inputs_embeds(
        self, *, input_ids: mx.array, audio_embeds: mx.array
    ) -> mx.array:
        inputs = self.language_model.model.embed_tokens(input_ids)
        if inputs.shape[:2] != audio_embeds.shape[:2]:
            raise ValueError("Audio8 text/audio clocks are not aligned")
        return inputs + audio_embeds.astype(inputs.dtype)

    def semantic_vad(self, hidden_states: mx.array) -> mx.array:
        if not self.semantic_vad_heads:
            raise ValueError("Audio8 checkpoint has no semantic VAD heads")
        logits = mx.stack(
            [head(hidden_states) for head in self.semantic_vad_heads], axis=1
        )
        return mx.softmax(logits.astype(mx.float32), axis=-1)

    def stream_session(
        self,
        *,
        language: str,
        frame_len: int = 4,
        target_delay_ms: int = 480,
        right_pad_text_tokens: int = 10,
        rolling_context_tokens: int = 375,
        rolling_trim_interval_ms: int = 3000,
    ):
        if self.tokenizer is None:
            raise ValueError("Audio8 tokenizer assets are not loaded")
        from .streaming import Audio8StreamSession

        return Audio8StreamSession(
            self,
            language=language,
            frame_len=frame_len,
            target_delay_ms=target_delay_ms,
            right_pad_text_tokens=right_pad_text_tokens,
            rolling_context_tokens=rolling_context_tokens,
            rolling_trim_interval_ms=rolling_trim_interval_ms,
        )

    def transcribe(
        self,
        audio: mx.array | np.ndarray,
        *,
        language: str,
        frame_len: int = 4,
        target_delay_ms: int = 480,
    ):
        session = self.stream_session(
            language=language,
            frame_len=frame_len,
            target_delay_ms=target_delay_ms,
        )
        session.feed(audio)
        session.finalize()
        return session.result()
