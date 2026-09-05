"""FireRedTTS3 CAM++ speaker embedding backed by the shared MLX implementation."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from ..dots_tts.speaker import (
    CAMPPlus,
    CAMPPlusConfig,
    SpeakerFrontend,
    _sinc_resample,
)
from .config import FireRedTTS3Config


class FireRedSpeakerEncoder:
    def __init__(
        self,
        model: CAMPPlus,
        *,
        max_audio_seconds: float = 60.0,
    ):
        self.model = model
        self.frontend = SpeakerFrontend(max_audio_seconds=max_audio_seconds)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "FireRedSpeakerEncoder":
        root = Path(model_dir)
        artifact = FireRedTTS3Config.from_dir(root)
        source = artifact.speaker
        model = CAMPPlus(
            CAMPPlusConfig(
                feature_dim=int(source["feature_dim"]),
                embedding_size=int(source["embedding_size"]),
                growth_rate=int(source["growth_rate"]),
                bottleneck_size=4,
                initial_channels=int(source["init_channels"]),
                block_layers=tuple(int(value) for value in source["block_layers"]),
            )
        )
        weights = mx.load(root / artifact.files["speaker"])
        model.load_weights(list(weights.items()), strict=True)
        return cls(model)

    def __call__(
        self,
        audio: np.ndarray | mx.array,
        *,
        sample_rate: int,
    ) -> mx.array:
        features, length = self.frontend.features(audio, sample_rate=sample_rate)
        return self.model(
            mx.array(features[None], dtype=mx.float32),
            lengths=mx.array([length], dtype=mx.int32),
        )


def resample_mono_audio(
    audio: np.ndarray | mx.array,
    *,
    source_rate: int,
    target_rate: int,
) -> mx.array:
    waveform = SpeakerFrontend._mono(audio)
    if waveform.size == 0:
        raise ValueError("FireRedTTS3 reference_audio must not be empty")
    if not np.isfinite(waveform).all():
        raise ValueError("FireRedTTS3 reference_audio contains non-finite values")
    return mx.array(
        _sinc_resample(waveform, source_rate, target_rate),
        dtype=mx.float32,
    )


__all__ = ["FireRedSpeakerEncoder", "resample_mono_audio"]
