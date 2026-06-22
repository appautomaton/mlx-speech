"""Configuration parsing for Qwen3-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Qwen3ASRAudioConfig:
    """Audio encoder configuration used by Qwen3-ASR."""

    d_model: int
    num_mel_bins: int
    encoder_layers: int
    encoder_attention_heads: int
    encoder_ffn_dim: int
    downsample_hidden_size: int
    output_dim: int
    max_source_positions: int
    n_window: int
    n_window_infer: int
    conv_chunksize: int
    activation_function: str = "gelu"
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    model_type: str = "qwen3_asr_audio_encoder"
    dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Qwen3ASRAudioConfig":
        consumed = {
            "d_model",
            "num_mel_bins",
            "encoder_layers",
            "encoder_attention_heads",
            "encoder_ffn_dim",
            "downsample_hidden_size",
            "output_dim",
            "max_source_positions",
            "n_window",
            "n_window_infer",
            "conv_chunksize",
            "activation_function",
            "dropout",
            "attention_dropout",
            "activation_dropout",
            "scale_embedding",
            "model_type",
            "dtype",
        }
        return cls(
            d_model=int(payload["d_model"]),
            num_mel_bins=int(payload["num_mel_bins"]),
            encoder_layers=int(payload["encoder_layers"]),
            encoder_attention_heads=int(payload["encoder_attention_heads"]),
            encoder_ffn_dim=int(payload["encoder_ffn_dim"]),
            downsample_hidden_size=int(payload["downsample_hidden_size"]),
            output_dim=int(payload["output_dim"]),
            max_source_positions=int(payload["max_source_positions"]),
            n_window=int(payload["n_window"]),
            n_window_infer=int(payload["n_window_infer"]),
            conv_chunksize=int(payload["conv_chunksize"]),
            activation_function=str(payload.get("activation_function", "gelu")),
            dropout=float(payload.get("dropout", 0.0)),
            attention_dropout=float(payload.get("attention_dropout", 0.0)),
            activation_dropout=float(payload.get("activation_dropout", 0.0)),
            scale_embedding=bool(payload.get("scale_embedding", False)),
            model_type=str(payload.get("model_type", "qwen3_asr_audio_encoder")),
            dtype=str(payload.get("dtype", "bfloat16")),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "d_model": self.d_model,
            "num_mel_bins": self.num_mel_bins,
            "encoder_layers": self.encoder_layers,
            "encoder_attention_heads": self.encoder_attention_heads,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "downsample_hidden_size": self.downsample_hidden_size,
            "output_dim": self.output_dim,
            "max_source_positions": self.max_source_positions,
            "n_window": self.n_window,
            "n_window_infer": self.n_window_infer,
            "conv_chunksize": self.conv_chunksize,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "attention_dropout": self.attention_dropout,
            "activation_dropout": self.activation_dropout,
            "scale_embedding": self.scale_embedding,
            "model_type": self.model_type,
            "dtype": self.dtype,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class Qwen3ASRTextConfig:
    """Qwen3 text decoder configuration used by Qwen3-ASR."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_cache: bool = True
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    model_type: str = "qwen3"
    dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Qwen3ASRTextConfig":
        consumed = {
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
            "rms_norm_eps",
            "rope_theta",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "use_cache",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "model_type",
            "dtype",
        }
        return cls(
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            head_dim=int(payload["head_dim"]),
            vocab_size=int(payload["vocab_size"]),
            max_position_embeddings=int(payload["max_position_embeddings"]),
            rms_norm_eps=float(payload["rms_norm_eps"]),
            rope_theta=float(payload["rope_theta"]),
            hidden_act=str(payload.get("hidden_act", "silu")),
            attention_bias=bool(payload.get("attention_bias", False)),
            attention_dropout=float(payload.get("attention_dropout", 0.0)),
            use_cache=bool(payload.get("use_cache", True)),
            bos_token_id=payload.get("bos_token_id"),
            eos_token_id=payload.get("eos_token_id"),
            pad_token_id=payload.get("pad_token_id"),
            model_type=str(payload.get("model_type", "qwen3")),
            dtype=str(payload.get("dtype", "bfloat16")),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "hidden_act": self.hidden_act,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "use_cache": self.use_cache,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "model_type": self.model_type,
            "dtype": self.dtype,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class Qwen3ASRThinkerConfig:
    """Combined audio/text thinker configuration for Qwen3-ASR."""

    audio_config: Qwen3ASRAudioConfig
    text_config: Qwen3ASRTextConfig
    audio_token_id: int
    audio_start_token_id: int
    audio_end_token_id: int
    model_type: str = "qwen3_asr"
    dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Qwen3ASRThinkerConfig":
        consumed = {
            "audio_config",
            "text_config",
            "audio_token_id",
            "audio_start_token_id",
            "audio_end_token_id",
            "model_type",
            "dtype",
        }
        return cls(
            audio_config=Qwen3ASRAudioConfig.from_dict(payload["audio_config"]),
            text_config=Qwen3ASRTextConfig.from_dict(payload["text_config"]),
            audio_token_id=int(payload["audio_token_id"]),
            audio_start_token_id=int(payload["audio_start_token_id"]),
            audio_end_token_id=int(payload["audio_end_token_id"]),
            model_type=str(payload.get("model_type", "qwen3_asr")),
            dtype=str(payload.get("dtype", "bfloat16")),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "audio_config": self.audio_config.to_dict(),
            "text_config": self.text_config.to_dict(),
            "audio_token_id": self.audio_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_id": self.audio_end_token_id,
            "model_type": self.model_type,
            "dtype": self.dtype,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class Qwen3ASRConfig:
    """Top-level Qwen3-ASR configuration."""

    thinker_config: Qwen3ASRThinkerConfig
    model_type: str = "qwen3_asr"
    architectures: tuple[str, ...] = ()
    support_languages: tuple[str, ...] = ()
    transformers_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Qwen3ASRConfig":
        model_type = str(payload.get("model_type", ""))
        if model_type != "qwen3_asr":
            raise ValueError(f"Expected model_type 'qwen3_asr', got {model_type!r}")
        consumed = {
            "thinker_config",
            "model_type",
            "architectures",
            "support_languages",
            "transformers_version",
        }
        return cls(
            thinker_config=Qwen3ASRThinkerConfig.from_dict(payload["thinker_config"]),
            model_type=model_type,
            architectures=tuple(str(x) for x in payload.get("architectures", ())),
            support_languages=tuple(str(x) for x in payload.get("support_languages", ())),
            transformers_version=(
                str(payload["transformers_version"])
                if payload.get("transformers_version") is not None
                else None
            ),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "thinker_config": self.thinker_config.to_dict(),
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "support_languages": list(self.support_languages),
            "transformers_version": self.transformers_version,
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Qwen3ASRConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Qwen3-ASR config not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @property
    def audio_config(self) -> Qwen3ASRAudioConfig:
        return self.thinker_config.audio_config

    @property
    def text_config(self) -> Qwen3ASRTextConfig:
        return self.thinker_config.text_config

    @property
    def audio_token_id(self) -> int:
        return self.thinker_config.audio_token_id

    @property
    def audio_start_token_id(self) -> int:
        return self.thinker_config.audio_start_token_id

    @property
    def audio_end_token_id(self) -> int:
        return self.thinker_config.audio_end_token_id
