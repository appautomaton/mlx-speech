"""Configuration helpers for MossTTSLocal."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Qwen3LanguageConfig:
    """Subset of the upstream Qwen3 config used by MossTTSLocal."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    model_type: str = "qwen3"
    tie_word_embeddings: bool = False
    rope_theta: float = 1_000_000.0
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def effective_head_dim(self) -> int:
        return self.head_dim or (self.hidden_size // self.num_attention_heads)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Qwen3LanguageConfig":
        field_names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        known_keys = field_names - {"extra"}
        kwargs = {key: payload[key] for key in known_keys if key in payload}
        extra = {key: value for key, value in payload.items() if key not in known_keys}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "head_dim": self.head_dim,
            "rms_norm_eps": self.rms_norm_eps,
            "attention_dropout": self.attention_dropout,
            "hidden_act": self.hidden_act,
            "model_type": self.model_type,
            "tie_word_embeddings": self.tie_word_embeddings,
            "rope_theta": self.rope_theta,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class MossTTSLocalConfig:
    """MLX-facing config view for the v0 MossTTSLocal target."""

    language_config: Qwen3LanguageConfig
    model_type: str = "moss_tts_local"
    initializer_range: float = 0.02
    n_vq: int = 32
    pad_token_id: int = 151643
    im_start_token_id: int = 151644
    im_end_token_id: int = 151645
    audio_vocab_size: int = 1024
    audio_user_slot_token_id: int = 151654
    audio_assistant_gen_slot_token_id: int = 151656
    audio_assistant_delay_slot_token_id: int = 151662
    audio_start_token_id: int = 151652
    audio_end_token_id: int = 151653
    audio_pad_code: int = 1024
    sampling_rate: int = 24000
    additional_mlp_ffn_hidden_size: int = 2048
    local_ffn_hidden_size: int = 8960
    local_hidden_size: int = 1536
    local_num_layers: int = 4
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def hidden_size(self) -> int:
        return self.language_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.language_config.vocab_size

    @property
    def channels(self) -> int:
        return 1 + self.n_vq

    @property
    def audio_embedding_vocab_size(self) -> int:
        return self.audio_vocab_size + 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MossTTSLocalConfig":
        if "language_config" not in payload or not isinstance(payload["language_config"], dict):
            raise ValueError("MossTTSLocal config requires a nested `language_config` dictionary.")

        field_names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        known_keys = field_names - {"language_config", "extra"}
        kwargs = {key: payload[key] for key in known_keys if key in payload}
        extra = {
            key: value
            for key, value in payload.items()
            if key not in known_keys and key != "language_config"
        }
        return cls(
            language_config=Qwen3LanguageConfig.from_dict(payload["language_config"]),
            extra=extra,
            **kwargs,
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "MossTTSLocalConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_type": self.model_type,
            "initializer_range": self.initializer_range,
            "n_vq": self.n_vq,
            "pad_token_id": self.pad_token_id,
            "im_start_token_id": self.im_start_token_id,
            "im_end_token_id": self.im_end_token_id,
            "audio_vocab_size": self.audio_vocab_size,
            "audio_user_slot_token_id": self.audio_user_slot_token_id,
            "audio_assistant_gen_slot_token_id": self.audio_assistant_gen_slot_token_id,
            "audio_assistant_delay_slot_token_id": self.audio_assistant_delay_slot_token_id,
            "audio_start_token_id": self.audio_start_token_id,
            "audio_end_token_id": self.audio_end_token_id,
            "audio_pad_code": self.audio_pad_code,
            "sampling_rate": self.sampling_rate,
            "additional_mlp_ffn_hidden_size": self.additional_mlp_ffn_hidden_size,
            "local_ffn_hidden_size": self.local_ffn_hidden_size,
            "local_hidden_size": self.local_hidden_size,
            "local_num_layers": self.local_num_layers,
            "language_config": self.language_config.to_dict(),
        }
        payload.update(self.extra)
        return payload
