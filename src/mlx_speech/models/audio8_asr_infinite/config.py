"""Configuration parsing for Audio8-ASR-Infinite."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Audio8AudioConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    num_mel_bins: int
    max_position_embeddings: int
    sliding_window: int
    rms_norm_eps: float
    rope_theta: float
    streaming_n_left_pad_tokens: int
    activation_function: str = "gelu"
    hidden_act: str = "silu"
    model_type: str = "voxtral_realtime_encoder"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Audio8AudioConfig":
        consumed = {
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "head_dim", "num_mel_bins",
            "max_position_embeddings", "sliding_window", "rms_norm_eps",
            "rope_parameters", "streaming_n_left_pad_tokens",
            "activation_function", "hidden_act", "model_type",
        }
        rope = payload.get("rope_parameters", {})
        return cls(
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            head_dim=int(payload["head_dim"]),
            num_mel_bins=int(payload["num_mel_bins"]),
            max_position_embeddings=int(payload["max_position_embeddings"]),
            sliding_window=int(payload["sliding_window"]),
            rms_norm_eps=float(payload["rms_norm_eps"]),
            rope_theta=float(rope.get("rope_theta", 1000000.0)),
            streaming_n_left_pad_tokens=int(payload["streaming_n_left_pad_tokens"]),
            activation_function=str(payload.get("activation_function", "gelu")),
            hidden_act=str(payload.get("hidden_act", "silu")),
            model_type=str(payload.get("model_type", "voxtral_realtime_encoder")),
            extra={key: value for key, value in payload.items() if key not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "head_dim": self.head_dim,
            "num_mel_bins": self.num_mel_bins,
            "max_position_embeddings": self.max_position_embeddings,
            "sliding_window": self.sliding_window,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"},
            "streaming_n_left_pad_tokens": self.streaming_n_left_pad_tokens,
            "activation_function": self.activation_function,
            "hidden_act": self.hidden_act,
            "model_type": self.model_type,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class Audio8TextConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    model_type: str = "qwen2"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Audio8TextConfig":
        consumed = {
            "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "vocab_size",
            "max_position_embeddings", "rms_norm_eps", "rope_parameters",
            "hidden_act", "tie_word_embeddings", "model_type", "head_dim",
        }
        rope = payload.get("rope_parameters", {})
        return cls(
            hidden_size=int(payload["hidden_size"]),
            intermediate_size=int(payload["intermediate_size"]),
            num_hidden_layers=int(payload["num_hidden_layers"]),
            num_attention_heads=int(payload["num_attention_heads"]),
            num_key_value_heads=int(payload["num_key_value_heads"]),
            vocab_size=int(payload["vocab_size"]),
            max_position_embeddings=int(payload["max_position_embeddings"]),
            rms_norm_eps=float(payload["rms_norm_eps"]),
            rope_theta=float(rope.get("rope_theta", 1000000.0)),
            hidden_act=str(payload.get("hidden_act", "silu")),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", True)),
            model_type=str(payload.get("model_type", "qwen2")),
            extra={key: value for key, value in payload.items() if key not in consumed},
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
            "rope_parameters": {"rope_theta": self.rope_theta, "rope_type": "default"},
            "hidden_act": self.hidden_act,
            "tie_word_embeddings": self.tie_word_embeddings,
            "model_type": self.model_type,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class Audio8Config:
    audio_config: Audio8AudioConfig
    text_config: Audio8TextConfig
    projection_size: int
    max_frame_len: int
    supported_frame_lens: tuple[int, ...]
    audio_length_per_tok: int
    audio_tower_frame_ms: int
    streaming_n_left_pad_tokens: int
    streaming_n_left_pad_tokens_by_frame_len: dict[int, int]
    num_delay_tokens_by_frame_len: dict[int, dict[int, int]]
    semantic_vad_horizons_seconds: tuple[float, ...]
    semantic_vad_num_classes: int
    use_frame_len_embedding: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    model_type: str = "audio8_asr_infinite"
    weight_format_version: int = 2
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Audio8Config":
        model_type = str(payload.get("model_type", ""))
        if model_type != "audio8_asr_infinite":
            raise ValueError(f"Expected model_type 'audio8_asr_infinite', got {model_type!r}")
        if int(payload.get("weight_format_version", 0)) != 2:
            raise ValueError("Audio8-ASR-Infinite requires weight_format_version=2")
        consumed = {
            "audio_config", "text_config", "projection_size", "max_frame_len",
            "supported_frame_lens", "frame_lens", "audio_length_per_tok",
            "audio_tower_frame_ms", "streaming_n_left_pad_tokens",
            "streaming_n_left_pad_tokens_by_frame_len",
            "num_delay_tokens_by_frame_len", "semantic_vad_horizons_seconds",
            "semantic_vad_num_classes", "use_frame_len_embedding",
            "bos_token_id", "eos_token_id", "pad_token_id", "model_type",
            "weight_format_version", "auto_map",
        }
        frame_lens = tuple(
            int(value)
            for value in payload.get("supported_frame_lens", payload.get("frame_lens", ()))
        )
        config = cls(
            audio_config=Audio8AudioConfig.from_dict(payload["audio_config"]),
            text_config=Audio8TextConfig.from_dict(payload["text_config"]),
            projection_size=int(payload["projection_size"]),
            max_frame_len=int(payload["max_frame_len"]),
            supported_frame_lens=frame_lens,
            audio_length_per_tok=int(payload["audio_length_per_tok"]),
            audio_tower_frame_ms=int(payload["audio_tower_frame_ms"]),
            streaming_n_left_pad_tokens=int(payload["streaming_n_left_pad_tokens"]),
            streaming_n_left_pad_tokens_by_frame_len={
                int(key): int(value)
                for key, value in payload["streaming_n_left_pad_tokens_by_frame_len"].items()
            },
            num_delay_tokens_by_frame_len={
                int(frame): {int(delay): int(count) for delay, count in values.items()}
                for frame, values in payload["num_delay_tokens_by_frame_len"].items()
            },
            semantic_vad_horizons_seconds=tuple(
                float(value) for value in payload.get("semantic_vad_horizons_seconds", ())
            ),
            semantic_vad_num_classes=int(payload.get("semantic_vad_num_classes", 8)),
            use_frame_len_embedding=bool(payload.get("use_frame_len_embedding", True)),
            bos_token_id=int(payload["bos_token_id"]),
            eos_token_id=int(payload["eos_token_id"]),
            pad_token_id=int(payload["pad_token_id"]),
            model_type=model_type,
            weight_format_version=2,
            extra={key: value for key, value in payload.items() if key not in consumed},
        )
        if config.text_config.model_type != "qwen2":
            raise ValueError("Audio8-ASR-Infinite currently requires a Qwen2 text decoder")
        if config.projection_size != config.max_frame_len * config.audio_config.hidden_size:
            raise ValueError("projection_size must equal max_frame_len * audio hidden_size")
        if set(config.supported_frame_lens) != set(config.streaming_n_left_pad_tokens_by_frame_len):
            raise ValueError("Every supported frame length needs a left-pad configuration")
        return config

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_type": self.model_type,
            "weight_format_version": self.weight_format_version,
            "audio_config": self.audio_config.to_dict(),
            "text_config": self.text_config.to_dict(),
            "projection_size": self.projection_size,
            "max_frame_len": self.max_frame_len,
            "supported_frame_lens": list(self.supported_frame_lens),
            "frame_lens": list(self.supported_frame_lens),
            "audio_length_per_tok": self.audio_length_per_tok,
            "audio_tower_frame_ms": self.audio_tower_frame_ms,
            "streaming_n_left_pad_tokens": self.streaming_n_left_pad_tokens,
            "streaming_n_left_pad_tokens_by_frame_len": {
                str(key): value for key, value in self.streaming_n_left_pad_tokens_by_frame_len.items()
            },
            "num_delay_tokens_by_frame_len": {
                str(frame): {str(delay): count for delay, count in values.items()}
                for frame, values in self.num_delay_tokens_by_frame_len.items()
            },
            "semantic_vad_horizons_seconds": list(self.semantic_vad_horizons_seconds),
            "semantic_vad_num_classes": self.semantic_vad_num_classes,
            "use_frame_len_embedding": self.use_frame_len_embedding,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Audio8Config":
        path = Path(model_dir) / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"Audio8 config not found: {path}")
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def clock_ms(self, frame_len: int) -> int:
        if frame_len not in self.supported_frame_lens:
            raise ValueError(f"Unsupported frame_len {frame_len}; choose {self.supported_frame_lens}")
        return self.audio_tower_frame_ms * frame_len

    def left_pad_tokens(self, frame_len: int) -> int:
        self.clock_ms(frame_len)
        return self.streaming_n_left_pad_tokens_by_frame_len[frame_len]

    def delay_tokens(self, frame_len: int, target_delay_ms: int) -> int:
        clock_ms = self.clock_ms(frame_len)
        if target_delay_ms <= 0 or target_delay_ms % clock_ms:
            raise ValueError(f"target_delay_ms must be a positive multiple of {clock_ms}")
        return target_delay_ms // clock_ms

    def is_optimized_operating_point(self, frame_len: int, target_delay_ms: int) -> bool:
        delay_tokens = self.delay_tokens(frame_len, target_delay_ms)
        return self.num_delay_tokens_by_frame_len.get(frame_len, {}).get(target_delay_ms) == delay_tokens
