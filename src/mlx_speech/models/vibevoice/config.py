"""Configuration for VibeVoice Large."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Qwen2LanguageConfig:
    """Qwen2 backbone config for VibeVoice Large."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    vocab_size: int
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    model_type: str = "qwen2"
    tie_word_embeddings: bool = False
    rope_theta: float = 1_000_000.0
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Qwen2LanguageConfig:
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        known = field_names - {"extra"}
        kwargs = {k: payload[k] for k in known if k in payload}
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
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
class VibeVoiceConvTokenizerConfig:
    """Shared config for the acoustic and semantic causal conv tokenizers."""

    vae_dim: int
    encoder_ratios: tuple[int, ...] = (8, 5, 5, 4, 2, 2)
    encoder_depths: str = "3-3-3-3-3-3-8"
    encoder_n_filters: int = 32
    decoder_ratios: tuple[int, ...] | None = None
    decoder_depths: str | None = None
    decoder_n_filters: int = 32
    channels: int = 1
    causal: bool = True
    fix_std: float = 0.5
    std_dist_type: str = "gaussian"
    mixer_layer: str = "depthwise_conv"
    layernorm: str = "RMSNorm"
    layernorm_eps: float = 1e-5
    conv_bias: bool = True
    layer_scale_init_value: float = 1e-6
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def parsed_encoder_depths(self) -> list[int]:
        return [int(d) for d in self.encoder_depths.split("-")]

    @property
    def parsed_decoder_depths(self) -> list[int]:
        if self.decoder_depths is not None:
            return [int(d) for d in self.decoder_depths.split("-")]
        return list(reversed(self.parsed_encoder_depths))

    @property
    def effective_decoder_ratios(self) -> tuple[int, ...]:
        if self.decoder_ratios is not None:
            return self.decoder_ratios
        return self.encoder_ratios

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VibeVoiceConvTokenizerConfig:
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        known = field_names - {"extra"}
        kwargs: dict[str, Any] = {}
        for k in known:
            if k not in payload:
                continue
            v = payload[k]
            if k in ("encoder_ratios", "decoder_ratios") and isinstance(v, list):
                v = tuple(v)
            kwargs[k] = v
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vae_dim": self.vae_dim,
            "encoder_ratios": list(self.encoder_ratios),
            "encoder_depths": self.encoder_depths,
            "encoder_n_filters": self.encoder_n_filters,
            "decoder_ratios": list(self.decoder_ratios) if self.decoder_ratios else None,
            "decoder_depths": self.decoder_depths,
            "decoder_n_filters": self.decoder_n_filters,
            "channels": self.channels,
            "causal": self.causal,
            "fix_std": self.fix_std,
            "std_dist_type": self.std_dist_type,
            "mixer_layer": self.mixer_layer,
            "layernorm": self.layernorm,
            "layernorm_eps": self.layernorm_eps,
            "conv_bias": self.conv_bias,
            "layer_scale_init_value": self.layer_scale_init_value,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class VibeVoiceDiffusionConfig:
    """Config for the diffusion prediction head."""

    hidden_size: int = 3584
    latent_size: int = 64
    head_layers: int = 4
    head_ffn_ratio: float = 3.0
    rms_norm_eps: float = 1e-5
    prediction_type: str = "v_prediction"
    ddpm_num_steps: int = 1000
    ddpm_num_inference_steps: int = 20
    ddpm_beta_schedule: str = "cosine"
    ddpm_batch_mul: int = 4
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VibeVoiceDiffusionConfig:
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        known = field_names - {"extra"}
        kwargs = {k: payload[k] for k in known if k in payload}
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(**kwargs, extra=extra)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "latent_size": self.latent_size,
            "head_layers": self.head_layers,
            "head_ffn_ratio": self.head_ffn_ratio,
            "rms_norm_eps": self.rms_norm_eps,
            "prediction_type": self.prediction_type,
            "ddpm_num_steps": self.ddpm_num_steps,
            "ddpm_num_inference_steps": self.ddpm_num_inference_steps,
            "ddpm_beta_schedule": self.ddpm_beta_schedule,
            "ddpm_batch_mul": self.ddpm_batch_mul,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class VibeVoiceConfig:
    """Top-level config for VibeVoice Large."""

    language_config: Qwen2LanguageConfig
    acoustic_tokenizer_config: VibeVoiceConvTokenizerConfig
    semantic_tokenizer_config: VibeVoiceConvTokenizerConfig
    diffusion_config: VibeVoiceDiffusionConfig
    model_type: str = "vibevoice"
    acoustic_vae_dim: int = 64
    semantic_vae_dim: int = 128
    sampling_rate: int = 24000
    speech_tok_compress_ratio: int = 3200
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def hidden_size(self) -> int:
        return self.language_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.language_config.vocab_size

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VibeVoiceConfig:
        raw = dict(payload)

        # Parse sub-configs — JSON keys differ from field names.
        language_config = Qwen2LanguageConfig.from_dict(raw.pop("decoder_config", {}))
        acoustic_config = VibeVoiceConvTokenizerConfig.from_dict(
            raw.pop("acoustic_tokenizer_config", {})
        )
        semantic_config = VibeVoiceConvTokenizerConfig.from_dict(
            raw.pop("semantic_tokenizer_config", {})
        )
        diffusion_config = VibeVoiceDiffusionConfig.from_dict(
            raw.pop("diffusion_head_config", {})
        )

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        known = field_names - {
            "extra",
            "language_config",
            "acoustic_tokenizer_config",
            "semantic_tokenizer_config",
            "diffusion_config",
        }
        # Handle the upstream typo: "acostic_vae_dim" → acoustic_vae_dim
        if "acostic_vae_dim" in raw and "acoustic_vae_dim" not in raw:
            raw["acoustic_vae_dim"] = raw.pop("acostic_vae_dim")

        kwargs = {k: raw.pop(k) for k in list(raw) if k in known}
        extra = {k: v for k, v in raw.items() if k not in known}

        return cls(
            language_config=language_config,
            acoustic_tokenizer_config=acoustic_config,
            semantic_tokenizer_config=semantic_config,
            diffusion_config=diffusion_config,
            **kwargs,
            extra=extra,
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> VibeVoiceConfig:
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_type": self.model_type,
            "decoder_config": self.language_config.to_dict(),
            "acoustic_tokenizer_config": self.acoustic_tokenizer_config.to_dict(),
            "semantic_tokenizer_config": self.semantic_tokenizer_config.to_dict(),
            "diffusion_head_config": self.diffusion_config.to_dict(),
            "acoustic_vae_dim": self.acoustic_vae_dim,
            "semantic_vae_dim": self.semantic_vae_dim,
            "sampling_rate": self.sampling_rate,
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
        }
        payload.update(self.extra)
        return payload
