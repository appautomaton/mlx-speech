"""Configuration schemas for the dots.tts MLX runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


def _positive_int(value: Any, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def _extra(payload: dict[str, Any], declared: set[str]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in declared}


@dataclass(frozen=True)
class DotsTTSTransformerConfig:
    num_layers: int
    num_heads: int
    hidden_size: int
    ffn_hidden_size: int
    modulation: bool
    qkv_bias: bool
    qk_norm: bool
    attn_dropout: float
    dropout: float
    norm_layer: str
    alibi_bias: bool
    rotary_bias: bool
    rotary_theta: float
    input_dim: int | None = None
    causal: bool | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSTransformerConfig":
        declared = {
            "num_layers",
            "num_heads",
            "hidden_size",
            "ffn_hidden_size",
            "modulation",
            "qkv_bias",
            "qk_norm",
            "attn_dropout",
            "dropout",
            "norm_layer",
            "alibi_bias",
            "rotary_bias",
            "rotary_theta",
            "input_dim",
            "causal",
        }
        missing = sorted(
            {
                "num_layers",
                "num_heads",
                "hidden_size",
                "ffn_hidden_size",
                "modulation",
                "qkv_bias",
                "qk_norm",
                "attn_dropout",
                "dropout",
                "norm_layer",
                "alibi_bias",
                "rotary_bias",
                "rotary_theta",
            }
            - payload.keys()
        )
        if missing:
            raise ValueError(f"transformer config is missing fields: {missing}")
        norm_layer = str(payload["norm_layer"])
        if norm_layer not in {"RMSNorm", "LayerNorm"}:
            raise ValueError(f"unsupported transformer norm_layer: {norm_layer}")
        return cls(
            num_layers=_positive_int(payload["num_layers"], "num_layers"),
            num_heads=_positive_int(payload["num_heads"], "num_heads"),
            hidden_size=_positive_int(payload["hidden_size"], "hidden_size"),
            ffn_hidden_size=_positive_int(
                payload["ffn_hidden_size"], "ffn_hidden_size"
            ),
            modulation=bool(payload["modulation"]),
            qkv_bias=bool(payload["qkv_bias"]),
            qk_norm=bool(payload["qk_norm"]),
            attn_dropout=float(payload["attn_dropout"]),
            dropout=float(payload["dropout"]),
            norm_layer=norm_layer,
            alibi_bias=bool(payload["alibi_bias"]),
            rotary_bias=bool(payload["rotary_bias"]),
            rotary_theta=float(payload["rotary_theta"]),
            input_dim=(
                _positive_int(payload["input_dim"], "input_dim")
                if payload.get("input_dim") is not None
                else None
            ),
            causal=(bool(payload["causal"]) if "causal" in payload else None),
            extra=_extra(payload, declared),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {**self.extra, **asdict(self)}
        payload.pop("extra")
        if self.input_dim is None:
            payload.pop("input_dim")
        if self.causal is None:
            payload.pop("causal")
        return payload


@dataclass(frozen=True)
class DotsTTSVocoderConfig:
    sample_rate: int
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    upsample_initial_channel: int
    resblock: str
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    downsample_rates: tuple[int, ...]
    downsample_channels: tuple[int, ...]
    activation: str
    snake_logscale: bool
    latent_dim: int
    causal: bool
    mi_num_layers: int
    causal_encoder: bool
    use_bias_at_final: bool
    use_tanh_at_final: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSVocoderConfig":
        declared = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        missing = sorted(declared - payload.keys())
        if missing:
            raise ValueError(f"vocoder config is missing fields: {missing}")
        upsample_rates = tuple(int(value) for value in payload["upsample_rates"])
        kernels = tuple(int(value) for value in payload["upsample_kernel_sizes"])
        if len(upsample_rates) != len(kernels):
            raise ValueError("vocoder upsample rates and kernels must have equal length")
        downsample_rates = tuple(int(value) for value in payload["downsample_rates"])
        channels = tuple(int(value) for value in payload["downsample_channels"])
        if len(channels) != len(downsample_rates) + 1:
            raise ValueError("vocoder downsample channels must be one longer than rates")
        return cls(
            sample_rate=_positive_int(payload["sample_rate"], "sample_rate"),
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=kernels,
            upsample_initial_channel=_positive_int(
                payload["upsample_initial_channel"], "upsample_initial_channel"
            ),
            resblock=str(payload["resblock"]),
            resblock_kernel_sizes=tuple(
                int(value) for value in payload["resblock_kernel_sizes"]
            ),
            resblock_dilation_sizes=tuple(
                tuple(int(value) for value in item)
                for item in payload["resblock_dilation_sizes"]
            ),
            downsample_rates=downsample_rates,
            downsample_channels=channels,
            activation=str(payload["activation"]),
            snake_logscale=bool(payload["snake_logscale"]),
            latent_dim=_positive_int(payload["latent_dim"], "latent_dim"),
            causal=bool(payload["causal"]),
            mi_num_layers=_positive_int(payload["mi_num_layers"], "mi_num_layers"),
            causal_encoder=bool(payload["causal_encoder"]),
            use_bias_at_final=bool(payload["use_bias_at_final"]),
            use_tanh_at_final=bool(payload["use_tanh_at_final"]),
            extra=_extra(payload, declared),
        )

    @property
    def hop_size(self) -> int:
        result = 1
        for rate in self.downsample_rates:
            result *= rate
        return result

    def to_dict(self) -> dict[str, Any]:
        payload = {**self.extra, **asdict(self)}
        payload.pop("extra")
        for key in (
            "upsample_rates",
            "upsample_kernel_sizes",
            "resblock_kernel_sizes",
            "downsample_rates",
            "downsample_channels",
        ):
            payload[key] = list(payload[key])
        payload["resblock_dilation_sizes"] = [
            list(item) for item in self.resblock_dilation_sizes
        ]
        return payload


@dataclass(frozen=True)
class DotsTTSMeanFlowConfig:
    enabled: bool = True
    use_duration_embedding: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSMeanFlowConfig":
        return cls(
            enabled=bool(payload.get("enabled", True)),
            use_duration_embedding=bool(payload.get("use_duration_embedding", True)),
            extra=_extra(payload, {"enabled", "use_duration_embedding"}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.extra,
            "enabled": self.enabled,
            "use_duration_embedding": self.use_duration_embedding,
        }


@dataclass(frozen=True)
class DotsTTSConfig:
    latent_dim: int
    patch_size: int
    patch_encoder: DotsTTSTransformerConfig
    dit: DotsTTSTransformerConfig
    vocoder: DotsTTSVocoderConfig
    model_type: str = "dots_tts"
    cfg_droprate: float = 0.2
    fm_sigma: float = 0.0
    xvec_drop_rate: float = 0.2
    campplus_embedding_size: int = 512
    xvec_max_audio_seconds: float = 10.0
    meanflow: DotsTTSMeanFlowConfig | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def mode(self) -> Literal["flow_matching", "meanflow"]:
        if self.meanflow is not None and self.meanflow.enabled:
            return "meanflow"
        return "flow_matching"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSConfig":
        declared = {
            "model_type",
            "latent_dim",
            "patch_size",
            "cfg_droprate",
            "PatchEncoder",
            "DiT",
            "vocoder",
            "fm_sigma",
            "xvec_drop_rate",
            "campplus_embedding_size",
            "xvec_max_audio_seconds",
            "meanflow",
        }
        model_type = str(payload.get("model_type", "dots_tts"))
        if model_type != "dots_tts":
            raise ValueError(f"unsupported dots.tts model_type: {model_type}")
        for name in ("latent_dim", "patch_size", "PatchEncoder", "DiT", "vocoder"):
            if name not in payload:
                raise ValueError(f"dots.tts config is missing {name}")
        meanflow = (
            DotsTTSMeanFlowConfig.from_dict(payload["meanflow"])
            if payload.get("meanflow") is not None
            else None
        )
        if meanflow is not None and meanflow.enabled and not meanflow.use_duration_embedding:
            raise ValueError("MeanFlow artifacts require duration embeddings")
        config = cls(
            model_type=model_type,
            latent_dim=_positive_int(payload["latent_dim"], "latent_dim"),
            patch_size=_positive_int(payload["patch_size"], "patch_size"),
            cfg_droprate=float(payload.get("cfg_droprate", 0.2)),
            patch_encoder=DotsTTSTransformerConfig.from_dict(payload["PatchEncoder"]),
            dit=DotsTTSTransformerConfig.from_dict(payload["DiT"]),
            vocoder=DotsTTSVocoderConfig.from_dict(payload["vocoder"]),
            fm_sigma=float(payload.get("fm_sigma", 0.0)),
            xvec_drop_rate=float(payload.get("xvec_drop_rate", 0.2)),
            campplus_embedding_size=_positive_int(
                payload.get("campplus_embedding_size", 512),
                "campplus_embedding_size",
            ),
            xvec_max_audio_seconds=float(payload.get("xvec_max_audio_seconds", 10.0)),
            meanflow=meanflow,
            extra=_extra(payload, declared),
        )
        if config.vocoder.latent_dim != config.latent_dim:
            raise ValueError("vocoder and model latent dimensions differ")
        if config.vocoder.sample_rate != 48_000:
            raise ValueError("dots.tts artifacts must decode at 48 kHz")
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "DotsTTSConfig":
        source = Path(path)
        if source.is_dir():
            source = source / "config.json"
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            **self.extra,
            "model_type": self.model_type,
            "latent_dim": self.latent_dim,
            "patch_size": self.patch_size,
            "cfg_droprate": self.cfg_droprate,
            "PatchEncoder": self.patch_encoder.to_dict(),
            "DiT": self.dit.to_dict(),
            "vocoder": self.vocoder.to_dict(),
            "fm_sigma": self.fm_sigma,
            "xvec_drop_rate": self.xvec_drop_rate,
            "campplus_embedding_size": self.campplus_embedding_size,
            "xvec_max_audio_seconds": self.xvec_max_audio_seconds,
        }
        if self.meanflow is not None:
            payload["meanflow"] = self.meanflow.to_dict()
        return payload


@dataclass(frozen=True)
class DotsTTSQwenConfig:
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    hidden_act: str = "silu"
    attention_dropout: float = 0.0
    model_type: str = "qwen2"
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DotsTTSQwenConfig":
        declared = {item.name for item in cls.__dataclass_fields__.values()} - {"extra"}
        missing = sorted(
            {
                "vocab_size",
                "max_position_embeddings",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "rms_norm_eps",
                "rope_theta",
                "tie_word_embeddings",
            }
            - payload.keys()
        )
        if missing:
            raise ValueError(f"Qwen config is missing fields: {missing}")
        model_type = str(payload.get("model_type", "qwen2"))
        if model_type != "qwen2":
            raise ValueError(f"unsupported dots.tts text model: {model_type}")
        config = cls(
            vocab_size=_positive_int(payload["vocab_size"], "vocab_size"),
            max_position_embeddings=_positive_int(
                payload["max_position_embeddings"], "max_position_embeddings"
            ),
            hidden_size=_positive_int(payload["hidden_size"], "hidden_size"),
            intermediate_size=_positive_int(
                payload["intermediate_size"], "intermediate_size"
            ),
            num_hidden_layers=_positive_int(
                payload["num_hidden_layers"], "num_hidden_layers"
            ),
            num_attention_heads=_positive_int(
                payload["num_attention_heads"], "num_attention_heads"
            ),
            num_key_value_heads=_positive_int(
                payload["num_key_value_heads"], "num_key_value_heads"
            ),
            rms_norm_eps=float(payload["rms_norm_eps"]),
            rope_theta=float(payload["rope_theta"]),
            tie_word_embeddings=bool(payload["tie_word_embeddings"]),
            hidden_act=str(payload.get("hidden_act", "silu")),
            attention_dropout=float(payload.get("attention_dropout", 0.0)),
            model_type=model_type,
            extra=_extra(payload, declared),
        )
        if config.num_attention_heads % config.num_key_value_heads:
            raise ValueError("Qwen attention heads must divide evenly by KV heads")
        if config.hidden_size % config.num_attention_heads:
            raise ValueError("Qwen hidden size must divide evenly by attention heads")
        if not config.tie_word_embeddings:
            raise ValueError("dots.tts requires tied Qwen embeddings")
        return config

    @classmethod
    def from_path(cls, path: str | Path) -> "DotsTTSQwenConfig":
        source = Path(path)
        if source.is_dir():
            source = source / "llm_config.json"
        return cls.from_dict(json.loads(source.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        payload = {**self.extra, **asdict(self)}
        payload.pop("extra")
        return payload


__all__ = [
    "DotsTTSConfig",
    "DotsTTSMeanFlowConfig",
    "DotsTTSQwenConfig",
    "DotsTTSTransformerConfig",
    "DotsTTSVocoderConfig",
]
