"""Configuration helpers for the Moss audio tokenizer path."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MossAudioTokenizerModuleConfig:
    """One encoder/decoder stage from the upstream config."""

    module_type: str
    params: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MossAudioTokenizerModuleConfig":
        payload = dict(payload)
        module_type = str(payload.pop("module_type"))
        return cls(module_type=module_type, params=payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {"module_type": self.module_type}
        payload.update(self.params)
        return payload


@dataclass(frozen=True)
class MossAudioTokenizerQuantizerConfig:
    """Quantizer configuration from the upstream codec config."""

    input_dim: int
    rvq_dim: int
    output_dim: int
    num_quantizers: int
    codebook_size: int
    codebook_dim: int
    quantizer_type: str = "rlfq"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MossAudioTokenizerQuantizerConfig":
        field_names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        known_keys = field_names - {"extra"}
        kwargs = {key: payload[key] for key in known_keys if key in payload}
        extra = {key: value for key, value in payload.items() if key not in known_keys}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "input_dim": self.input_dim,
            "rvq_dim": self.rvq_dim,
            "output_dim": self.output_dim,
            "num_quantizers": self.num_quantizers,
            "codebook_size": self.codebook_size,
            "codebook_dim": self.codebook_dim,
            "quantizer_type": self.quantizer_type,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class MossAudioTokenizerConfig:
    """MLX-facing config for the Cat audio tokenizer path."""

    sampling_rate: int = 24000
    downsample_rate: int = 1920
    causal_transformer_context_duration: float = 10.0
    encoder_kwargs: tuple[MossAudioTokenizerModuleConfig, ...] = ()
    decoder_kwargs: tuple[MossAudioTokenizerModuleConfig, ...] = ()
    quantizer_type: str = "rlfq"
    quantizer_kwargs: MossAudioTokenizerQuantizerConfig | None = None
    version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_quantizers(self) -> int:
        if self.quantizer_kwargs is None:
            return 32
        return self.quantizer_kwargs.num_quantizers

    @property
    def frame_rate(self) -> float:
        return self.sampling_rate / self.downsample_rate

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MossAudioTokenizerConfig":
        payload = dict(payload)
        encoder_kwargs = tuple(
            MossAudioTokenizerModuleConfig.from_dict(item)
            for item in payload.get("encoder_kwargs", [])
        )
        decoder_kwargs = tuple(
            MossAudioTokenizerModuleConfig.from_dict(item)
            for item in payload.get("decoder_kwargs", [])
        )
        quantizer_kwargs_payload = payload.get("quantizer_kwargs")
        quantizer_kwargs = (
            MossAudioTokenizerQuantizerConfig.from_dict(quantizer_kwargs_payload)
            if isinstance(quantizer_kwargs_payload, dict)
            else None
        )

        known_keys = {
            "version",
            "sampling_rate",
            "sample_rate",
            "downsample_rate",
            "causal_transformer_context_duration",
            "encoder_kwargs",
            "decoder_kwargs",
            "quantizer_type",
            "quantizer_kwargs",
        }
        extra = {key: value for key, value in payload.items() if key not in known_keys}

        return cls(
            version=payload.get("version"),
            sampling_rate=int(payload.get("sampling_rate", payload.get("sample_rate", 24000))),
            downsample_rate=int(payload["downsample_rate"]),
            causal_transformer_context_duration=float(payload["causal_transformer_context_duration"]),
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            quantizer_type=str(payload.get("quantizer_type", "rlfq")),
            quantizer_kwargs=quantizer_kwargs,
            extra=extra,
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "MossAudioTokenizerConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_type": "moss-audio-tokenizer",
            "version": self.version,
            "sampling_rate": self.sampling_rate,
            "sample_rate": self.sampling_rate,
            "downsample_rate": self.downsample_rate,
            "causal_transformer_context_duration": self.causal_transformer_context_duration,
            "encoder_kwargs": [item.to_dict() for item in self.encoder_kwargs],
            "decoder_kwargs": [item.to_dict() for item in self.decoder_kwargs],
            "quantizer_type": self.quantizer_type,
            "quantizer_kwargs": (
                self.quantizer_kwargs.to_dict() if self.quantizer_kwargs is not None else None
            ),
        }
        payload.update(self.extra)
        return payload
