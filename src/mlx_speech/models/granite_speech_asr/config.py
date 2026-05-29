"""Configuration for IBM Granite Speech 4.0 1B ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GraniteSpeechEncoderConfig:
    """Conformer CTC encoder configuration from upstream config.json."""

    input_dim: int = 160
    hidden_dim: int = 1024
    output_dim: int = 348
    num_layers: int = 16
    num_heads: int = 8
    dim_head: int = 128
    feedforward_mult: int = 4
    conv_expansion_factor: int = 2
    conv_kernel_size: int = 15
    context_size: int = 200
    max_pos_emb: int = 512
    dropout: float = 0.1
    model_type: str = "granite_speech_encoder"
    torch_dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraniteSpeechEncoderConfig":
        consumed = {
            "input_dim",
            "hidden_dim",
            "output_dim",
            "num_layers",
            "num_heads",
            "dim_head",
            "feedforward_mult",
            "conv_expansion_factor",
            "conv_kernel_size",
            "context_size",
            "max_pos_emb",
            "dropout",
            "model_type",
            "torch_dtype",
        }
        return cls(
            input_dim=int(payload.get("input_dim", 160)),
            hidden_dim=int(payload.get("hidden_dim", 1024)),
            output_dim=int(payload.get("output_dim", 348)),
            num_layers=int(payload.get("num_layers", 16)),
            num_heads=int(payload.get("num_heads", 8)),
            dim_head=int(payload.get("dim_head", 128)),
            feedforward_mult=int(payload.get("feedforward_mult", 4)),
            conv_expansion_factor=int(payload.get("conv_expansion_factor", 2)),
            conv_kernel_size=int(payload.get("conv_kernel_size", 15)),
            context_size=int(payload.get("context_size", 200)),
            max_pos_emb=int(payload.get("max_pos_emb", 512)),
            dropout=float(payload.get("dropout", 0.1)),
            model_type=str(payload.get("model_type", "granite_speech_encoder")),
            torch_dtype=str(payload.get("torch_dtype", "bfloat16")),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dim_head": self.dim_head,
            "feedforward_mult": self.feedforward_mult,
            "conv_expansion_factor": self.conv_expansion_factor,
            "conv_kernel_size": self.conv_kernel_size,
            "context_size": self.context_size,
            "max_pos_emb": self.max_pos_emb,
            "dropout": self.dropout,
            "model_type": self.model_type,
            "torch_dtype": self.torch_dtype,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class GraniteSpeechProjectorConfig:
    """BLIP-2 QFormer projector configuration."""

    hidden_size: int = 1024
    encoder_hidden_size: int = 1024
    intermediate_size: int = 4096
    num_hidden_layers: int = 2
    num_attention_heads: int = 16
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    cross_attention_frequency: int = 1
    max_position_embeddings: int = 2048
    use_qformer_text_input: bool = False
    vocab_size: int = 30522
    initializer_range: float = 0.02
    position_embedding_type: str = "absolute"
    model_type: str = "blip_2_qformer"
    torch_dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraniteSpeechProjectorConfig":
        consumed = {
            "hidden_size",
            "encoder_hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "layer_norm_eps",
            "hidden_act",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "cross_attention_frequency",
            "max_position_embeddings",
            "use_qformer_text_input",
            "vocab_size",
            "initializer_range",
            "position_embedding_type",
            "model_type",
            "torch_dtype",
        }
        return cls(
            hidden_size=int(payload.get("hidden_size", 1024)),
            encoder_hidden_size=int(payload.get("encoder_hidden_size", 1024)),
            intermediate_size=int(payload.get("intermediate_size", 4096)),
            num_hidden_layers=int(payload.get("num_hidden_layers", 2)),
            num_attention_heads=int(payload.get("num_attention_heads", 16)),
            layer_norm_eps=float(payload.get("layer_norm_eps", 1e-12)),
            hidden_act=str(payload.get("hidden_act", "gelu")),
            hidden_dropout_prob=float(payload.get("hidden_dropout_prob", 0.1)),
            attention_probs_dropout_prob=float(payload.get("attention_probs_dropout_prob", 0.1)),
            cross_attention_frequency=int(payload.get("cross_attention_frequency", 1)),
            max_position_embeddings=int(payload.get("max_position_embeddings", 2048)),
            use_qformer_text_input=bool(payload.get("use_qformer_text_input", False)),
            vocab_size=int(payload.get("vocab_size", 30522)),
            initializer_range=float(payload.get("initializer_range", 0.02)),
            position_embedding_type=str(payload.get("position_embedding_type", "absolute")),
            model_type=str(payload.get("model_type", "blip_2_qformer")),
            torch_dtype=str(payload.get("torch_dtype", "bfloat16")),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "encoder_hidden_size": self.encoder_hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "layer_norm_eps": self.layer_norm_eps,
            "hidden_act": self.hidden_act,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "attention_probs_dropout_prob": self.attention_probs_dropout_prob,
            "cross_attention_frequency": self.cross_attention_frequency,
            "max_position_embeddings": self.max_position_embeddings,
            "use_qformer_text_input": self.use_qformer_text_input,
            "vocab_size": self.vocab_size,
            "initializer_range": self.initializer_range,
            "position_embedding_type": self.position_embedding_type,
            "model_type": self.model_type,
            "torch_dtype": self.torch_dtype,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class GraniteSpeechTextConfig:
    """Granite causal LM configuration used by the speech model."""

    vocab_size: int = 100353
    hidden_size: int = 2048
    intermediate_size: int = 4096
    num_hidden_layers: int = 40
    num_attention_heads: int = 16
    num_key_value_heads: int = 4
    max_position_embeddings: int = 4096
    hidden_act: str = "silu"
    attention_bias: bool = False
    attention_dropout: float = 0.0
    mlp_bias: bool = False
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_scaling: dict[str, Any] | None = None
    rope_parameters: dict[str, Any] = field(default_factory=lambda: {"rope_theta": 10000, "rope_type": "default"})
    use_cache: bool = True
    bos_token_id: int = 100257
    eos_token_id: int = 100257
    pad_token_id: int = 100256
    embedding_multiplier: float = 12.0
    attention_multiplier: float = 0.0078125
    residual_multiplier: float = 0.22
    logits_scaling: float = 8.0
    initializer_range: float = 0.1
    model_type: str = "granite"
    dtype: str = "float32"
    torch_dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraniteSpeechTextConfig":
        consumed = {
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "mlp_bias",
            "rms_norm_eps",
            "rope_theta",
            "rope_scaling",
            "rope_parameters",
            "use_cache",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "embedding_multiplier",
            "attention_multiplier",
            "residual_multiplier",
            "logits_scaling",
            "initializer_range",
            "model_type",
            "dtype",
            "torch_dtype",
        }
        return cls(
            vocab_size=int(payload.get("vocab_size", 100353)),
            hidden_size=int(payload.get("hidden_size", 2048)),
            intermediate_size=int(payload.get("intermediate_size", 4096)),
            num_hidden_layers=int(payload.get("num_hidden_layers", 40)),
            num_attention_heads=int(payload.get("num_attention_heads", 16)),
            num_key_value_heads=int(payload.get("num_key_value_heads", 4)),
            max_position_embeddings=int(payload.get("max_position_embeddings", 4096)),
            hidden_act=str(payload.get("hidden_act", "silu")),
            attention_bias=bool(payload.get("attention_bias", False)),
            attention_dropout=float(payload.get("attention_dropout", 0.0)),
            mlp_bias=bool(payload.get("mlp_bias", False)),
            rms_norm_eps=float(payload.get("rms_norm_eps", 1e-5)),
            rope_theta=float(payload.get("rope_theta", 10000.0)),
            rope_scaling=payload.get("rope_scaling"),
            rope_parameters=dict(payload.get("rope_parameters") or {"rope_theta": 10000, "rope_type": "default"}),
            use_cache=bool(payload.get("use_cache", True)),
            bos_token_id=int(payload.get("bos_token_id", 100257)),
            eos_token_id=int(payload.get("eos_token_id", 100257)),
            pad_token_id=int(payload.get("pad_token_id", 100256)),
            embedding_multiplier=float(payload.get("embedding_multiplier", 12.0)),
            attention_multiplier=float(payload.get("attention_multiplier", 0.0078125)),
            residual_multiplier=float(payload.get("residual_multiplier", 0.22)),
            logits_scaling=float(payload.get("logits_scaling", 8.0)),
            initializer_range=float(payload.get("initializer_range", 0.1)),
            model_type=str(payload.get("model_type", "granite")),
            dtype=str(payload.get("dtype", "float32")),
            torch_dtype=str(payload.get("torch_dtype", "bfloat16")),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "max_position_embeddings": self.max_position_embeddings,
            "hidden_act": self.hidden_act,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
            "mlp_bias": self.mlp_bias,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "rope_parameters": self.rope_parameters,
            "use_cache": self.use_cache,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token_id": self.pad_token_id,
            "embedding_multiplier": self.embedding_multiplier,
            "attention_multiplier": self.attention_multiplier,
            "residual_multiplier": self.residual_multiplier,
            "logits_scaling": self.logits_scaling,
            "initializer_range": self.initializer_range,
            "model_type": self.model_type,
            "dtype": self.dtype,
            "torch_dtype": self.torch_dtype,
        }
        payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class GraniteSpeechConfig:
    """Top-level Granite Speech ASR configuration."""

    encoder: GraniteSpeechEncoderConfig = field(default_factory=GraniteSpeechEncoderConfig)
    projector: GraniteSpeechProjectorConfig = field(default_factory=GraniteSpeechProjectorConfig)
    text: GraniteSpeechTextConfig = field(default_factory=GraniteSpeechTextConfig)
    model_type: str = "granite_speech"
    audio_token_index: int = 100352
    downsample_rate: int = 5
    window_size: int = 15
    has_lora_adapter: bool = False
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    torch_dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraniteSpeechConfig":
        consumed = {
            "encoder_config",
            "projector_config",
            "text_config",
            "model_type",
            "audio_token_index",
            "downsample_rate",
            "window_size",
            "has_lora_adapter",
            "tie_word_embeddings",
            "initializer_range",
            "torch_dtype",
        }
        return cls(
            encoder=GraniteSpeechEncoderConfig.from_dict(payload["encoder_config"]),
            projector=GraniteSpeechProjectorConfig.from_dict(payload["projector_config"]),
            text=GraniteSpeechTextConfig.from_dict(payload["text_config"]),
            model_type=str(payload.get("model_type", "granite_speech")),
            audio_token_index=int(payload.get("audio_token_index", 100352)),
            downsample_rate=int(payload.get("downsample_rate", 5)),
            window_size=int(payload.get("window_size", 15)),
            has_lora_adapter=bool(payload.get("has_lora_adapter", False)),
            tie_word_embeddings=bool(payload.get("tie_word_embeddings", False)),
            initializer_range=float(payload.get("initializer_range", 0.02)),
            torch_dtype=str(payload.get("torch_dtype", "bfloat16")),
            extra={k: v for k, v in payload.items() if k not in consumed},
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "GraniteSpeechConfig":
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_type": self.model_type,
            "audio_token_index": self.audio_token_index,
            "downsample_rate": self.downsample_rate,
            "window_size": self.window_size,
            "has_lora_adapter": self.has_lora_adapter,
            "tie_word_embeddings": self.tie_word_embeddings,
            "initializer_range": self.initializer_range,
            "torch_dtype": self.torch_dtype,
            "encoder_config": self.encoder.to_dict(),
            "projector_config": self.projector.to_dict(),
            "text_config": self.text.to_dict(),
        }
        payload.update(self.extra)
        return payload
