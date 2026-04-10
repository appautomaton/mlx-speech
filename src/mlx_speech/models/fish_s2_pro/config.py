from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


_MISSING = object()


def _pick_field(payload: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    return _MISSING


def _get_field(payload: dict[str, Any], default: Any, *names: str) -> Any:
    value = _pick_field(payload, *names)
    return default if value is _MISSING else value


@dataclass
class FishTextConfig:
    vocab_size: int = 155776
    n_layer: int = 36
    n_head: int = 32
    n_local_heads: int = 8
    head_dim: int = 128
    dim: int = 2560
    intermediate_size: int = 9728
    rope_base: float = 1_000_000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 32768
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FishTextConfig":
        return cls(
            vocab_size=int(_get_field(payload, cls.vocab_size, "vocab_size")),
            n_layer=int(
                _get_field(payload, cls.n_layer, "n_layer", "num_hidden_layers")
            ),
            n_head=int(
                _get_field(payload, cls.n_head, "n_head", "num_attention_heads")
            ),
            n_local_heads=int(
                _get_field(
                    payload,
                    cls.n_local_heads,
                    "n_local_heads",
                    "num_key_value_heads",
                )
            ),
            head_dim=int(_get_field(payload, cls.head_dim, "head_dim")),
            dim=int(_get_field(payload, cls.dim, "dim", "hidden_size")),
            intermediate_size=int(
                _get_field(payload, cls.intermediate_size, "intermediate_size")
            ),
            rope_base=float(_get_field(payload, cls.rope_base, "rope_base")),
            norm_eps=float(
                _get_field(payload, cls.norm_eps, "norm_eps", "rms_norm_eps")
            ),
            max_seq_len=int(
                _get_field(
                    payload,
                    cls.max_seq_len,
                    "max_seq_len",
                    "max_position_embeddings",
                )
            ),
            attention_qkv_bias=bool(
                _get_field(payload, cls.attention_qkv_bias, "attention_qkv_bias")
            ),
            attention_o_bias=bool(
                _get_field(payload, cls.attention_o_bias, "attention_o_bias")
            ),
            attention_qk_norm=bool(
                _get_field(payload, cls.attention_qk_norm, "attention_qk_norm")
            ),
        )


@dataclass
class FishAudioDecoderConfig:
    vocab_size: int = 4096
    n_layer: int = 4
    n_head: int = 32
    n_local_heads: int = 8
    head_dim: int = 128
    dim: int = 2560
    intermediate_size: int = 9728
    rope_base: float = 1_000_000.0
    norm_eps: float = 1e-6
    max_seq_len: int = 11
    attention_qkv_bias: bool = False
    attention_o_bias: bool = False
    attention_qk_norm: bool = False
    text_dim: int = 2560
    num_codebooks: int = 10

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FishAudioDecoderConfig":
        return cls(
            vocab_size=int(_get_field(payload, cls.vocab_size, "vocab_size")),
            n_layer=int(
                _get_field(payload, cls.n_layer, "n_layer", "num_hidden_layers")
            ),
            n_head=int(
                _get_field(payload, cls.n_head, "n_head", "num_attention_heads")
            ),
            n_local_heads=int(
                _get_field(
                    payload,
                    cls.n_local_heads,
                    "n_local_heads",
                    "num_key_value_heads",
                )
            ),
            head_dim=int(_get_field(payload, cls.head_dim, "head_dim")),
            dim=int(_get_field(payload, cls.dim, "dim", "hidden_size")),
            intermediate_size=int(
                _get_field(payload, cls.intermediate_size, "intermediate_size")
            ),
            rope_base=float(_get_field(payload, cls.rope_base, "rope_base")),
            norm_eps=float(
                _get_field(payload, cls.norm_eps, "norm_eps", "rms_norm_eps")
            ),
            max_seq_len=int(
                _get_field(
                    payload,
                    cls.max_seq_len,
                    "max_seq_len",
                    "max_position_embeddings",
                )
            ),
            attention_qkv_bias=bool(
                _get_field(payload, cls.attention_qkv_bias, "attention_qkv_bias")
            ),
            attention_o_bias=bool(
                _get_field(payload, cls.attention_o_bias, "attention_o_bias")
            ),
            attention_qk_norm=bool(
                _get_field(payload, cls.attention_qk_norm, "attention_qk_norm")
            ),
            text_dim=int(_get_field(payload, cls.text_dim, "text_dim")),
            num_codebooks=int(_get_field(payload, cls.num_codebooks, "num_codebooks")),
        )


@dataclass
class FishS2ProConfig:
    model_type: str = "fish_qwen3_omni"
    dtype: str = "bfloat16"
    sample_rate: int = 44100
    pad_token_id: int = 151669
    eos_token_id: int = 151645
    audio_pad_token_id: int = 151677
    semantic_start_token_id: int = 151678
    semantic_end_token_id: int = 155773
    model_dir: str = "models/fish_s2_pro/original"
    text_config: FishTextConfig = field(default_factory=FishTextConfig)
    audio_decoder_config: FishAudioDecoderConfig = field(
        default_factory=FishAudioDecoderConfig
    )

    @classmethod
    def from_dict(
        cls, payload: dict[str, Any], *, model_dir: str | None = None
    ) -> "FishS2ProConfig":
        return cls(
            model_type=str(_get_field(payload, cls.model_type, "model_type")),
            dtype=str(_get_field(payload, cls.dtype, "dtype")),
            sample_rate=int(_get_field(payload, cls.sample_rate, "sample_rate")),
            pad_token_id=int(_get_field(payload, cls.pad_token_id, "pad_token_id")),
            eos_token_id=int(_get_field(payload, cls.eos_token_id, "eos_token_id")),
            audio_pad_token_id=int(
                _get_field(payload, cls.audio_pad_token_id, "audio_pad_token_id")
            ),
            semantic_start_token_id=int(
                _get_field(
                    payload, cls.semantic_start_token_id, "semantic_start_token_id"
                )
            ),
            semantic_end_token_id=int(
                _get_field(payload, cls.semantic_end_token_id, "semantic_end_token_id")
            ),
            model_dir=str(model_dir)
            if model_dir is not None
            else str(_get_field(payload, cls.model_dir, "model_dir")),
            text_config=FishTextConfig.from_dict(payload.get("text_config", {})),
            audio_decoder_config=FishAudioDecoderConfig.from_dict(
                payload.get("audio_decoder_config", {})
            ),
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "FishS2ProConfig":
        resolved = Path(model_dir)
        with (resolved / "config.json").open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f), model_dir=str(resolved))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["text_config"] = asdict(self.text_config)
        payload["audio_decoder_config"] = asdict(self.audio_decoder_config)
        return payload
