"""Configuration helpers for LongCat AudioDiT 3.5B."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int
    group_size: int
    mode: str = "affine"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuantizationConfig":
        return cls(
            bits=int(payload["bits"]),
            group_size=int(payload["group_size"]),
            mode=str(payload.get("mode", "affine")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True)
class LongCatTextEncoderConfig:
    d_model: int = 768
    d_ff: int = 2048
    d_kv: int = 64
    num_heads: int = 12
    num_layers: int = 12
    num_decoder_layers: int = 12
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    vocab_size: int = 256384
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongCatTextEncoderConfig":
        raw = dict(payload)
        known = {field.name for field in cls.__dataclass_fields__.values()} - {"extra"}  # type: ignore[attr-defined]
        kwargs = {key: raw[key] for key in known if key in raw}
        extra = {key: value for key, value in raw.items() if key not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "d_model": self.d_model,
            "d_ff": self.d_ff,
            "d_kv": self.d_kv,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "dropout_rate": self.dropout_rate,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "relative_attention_num_buckets": self.relative_attention_num_buckets,
            "relative_attention_max_distance": self.relative_attention_max_distance,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class LongCatVaeConfig:
    in_channels: int = 1
    channels: int = 128
    c_mults: tuple[int, ...] = (1, 2, 4, 8, 16)
    strides: tuple[int, ...] = (2, 4, 4, 8, 8)
    latent_dim: int = 64
    encoder_latent_dim: int = 128
    use_snake: bool = True
    downsample_shortcut: str = "averaging"
    upsample_shortcut: str = "duplicating"
    out_shortcut: str = "averaging"
    in_shortcut: str = "duplicating"
    final_tanh: bool = False
    downsampling_ratio: int = 2048
    sample_rate: int = 24000
    scale: float = 0.71
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongCatVaeConfig":
        raw = dict(payload)
        if "c_mults" in raw:
            raw["c_mults"] = tuple(int(value) for value in raw["c_mults"])
        if "strides" in raw:
            raw["strides"] = tuple(int(value) for value in raw["strides"])
        known = {field.name for field in cls.__dataclass_fields__.values()} - {"extra"}  # type: ignore[attr-defined]
        kwargs = {key: raw[key] for key in known if key in raw}
        extra = {key: value for key, value in raw.items() if key not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "in_channels": self.in_channels,
            "channels": self.channels,
            "c_mults": list(self.c_mults),
            "strides": list(self.strides),
            "latent_dim": self.latent_dim,
            "encoder_latent_dim": self.encoder_latent_dim,
            "use_snake": self.use_snake,
            "downsample_shortcut": self.downsample_shortcut,
            "upsample_shortcut": self.upsample_shortcut,
            "out_shortcut": self.out_shortcut,
            "in_shortcut": self.in_shortcut,
            "final_tanh": self.final_tanh,
            "downsampling_ratio": self.downsampling_ratio,
            "sample_rate": self.sample_rate,
            "scale": self.scale,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class LongCatAudioDiTConfig:
    model_type: str = "audiodit"
    sampling_rate: int = 24000
    latent_hop: int = 2048
    latent_dim: int = 64
    max_wav_duration: int = 60
    dit_dim: int = 2560
    dit_depth: int = 32
    dit_heads: int = 32
    dit_ff_mult: float = 3.6
    dit_bias: bool = True
    dit_eps: float = 1e-6
    dit_dropout: float = 0.0
    dit_adaln_type: str = "global"
    dit_adaln_use_text_cond: bool = True
    dit_cross_attn: bool = True
    dit_cross_attn_norm: bool = False
    dit_long_skip: bool = True
    dit_qk_norm: bool = True
    dit_text_conv: bool = True
    dit_text_dim: int = 768
    dit_use_latent_condition: bool = True
    repa_dit_layer: int = 8
    sigma: float = 0.0
    text_encoder_model: str = "google/umt5-base"
    text_norm_feat: bool = True
    text_add_embed: bool = True
    text_encoder_config: LongCatTextEncoderConfig = field(
        default_factory=LongCatTextEncoderConfig
    )
    vae_config: LongCatVaeConfig = field(default_factory=LongCatVaeConfig)
    quantization: QuantizationConfig | None = None
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LongCatAudioDiTConfig":
        raw = dict(payload)
        known = {field.name for field in cls.__dataclass_fields__.values()} - {
            "text_encoder_config",
            "vae_config",
            "quantization",
            "extra",
        }  # type: ignore[attr-defined]
        kwargs = {key: raw[key] for key in known if key in raw}
        extra = {
            key: value
            for key, value in raw.items()
            if key not in known | {"text_encoder_config", "vae_config", "quantization"}
        }
        quantization = raw.get("quantization")
        return cls(
            **kwargs,
            text_encoder_config=LongCatTextEncoderConfig.from_dict(
                raw.get("text_encoder_config", {})
            ),
            vae_config=LongCatVaeConfig.from_dict(raw.get("vae_config", {})),
            quantization=QuantizationConfig.from_dict(quantization)
            if isinstance(quantization, dict)
            else None,
            extra=extra,
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "LongCatAudioDiTConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"LongCat config not found: {config_path}")
        with config_path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_type": self.model_type,
            "sampling_rate": self.sampling_rate,
            "latent_hop": self.latent_hop,
            "latent_dim": self.latent_dim,
            "max_wav_duration": self.max_wav_duration,
            "dit_dim": self.dit_dim,
            "dit_depth": self.dit_depth,
            "dit_heads": self.dit_heads,
            "dit_ff_mult": self.dit_ff_mult,
            "dit_bias": self.dit_bias,
            "dit_eps": self.dit_eps,
            "dit_dropout": self.dit_dropout,
            "dit_adaln_type": self.dit_adaln_type,
            "dit_adaln_use_text_cond": self.dit_adaln_use_text_cond,
            "dit_cross_attn": self.dit_cross_attn,
            "dit_cross_attn_norm": self.dit_cross_attn_norm,
            "dit_long_skip": self.dit_long_skip,
            "dit_qk_norm": self.dit_qk_norm,
            "dit_text_conv": self.dit_text_conv,
            "dit_text_dim": self.dit_text_dim,
            "dit_use_latent_condition": self.dit_use_latent_condition,
            "repa_dit_layer": self.repa_dit_layer,
            "sigma": self.sigma,
            "text_encoder_model": self.text_encoder_model,
            "text_norm_feat": self.text_norm_feat,
            "text_add_embed": self.text_add_embed,
            "text_encoder_config": self.text_encoder_config.to_dict(),
            "vae_config": self.vae_config.to_dict(),
        }
        if self.quantization is not None:
            payload["quantization"] = self.quantization.to_dict()
        payload.update(self.extra)
        return payload
