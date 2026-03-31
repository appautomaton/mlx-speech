"""Configuration for CohereAsr (cohere-transcribe-03-2026)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ParakeetEncoderConfig:
    """Fast-Conformer encoder configuration (subset of ParakeetEncoderConfig)."""

    hidden_size: int = 1280
    num_hidden_layers: int = 48
    num_attention_heads: int = 8
    intermediate_size: int = 5120
    hidden_act: str = "silu"
    conv_kernel_size: int = 9
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_mel_bins: int = 128
    subsampling_conv_kernel_size: int = 3
    subsampling_conv_stride: int = 2
    max_position_embeddings: int = 5000
    scale_input: bool = False
    attention_bias: bool = True
    convolution_bias: bool = True
    dropout: float = 0.0
    dropout_positions: float = 0.0
    layerdrop: float = 0.0
    activation_dropout: float = 0.0
    attention_dropout: float = 0.0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_subsampling_layers(self) -> int:
        import math
        return int(math.log2(self.subsampling_factor))

    @classmethod
    def from_nemo_dict(cls, enc: dict[str, Any]) -> "ParakeetEncoderConfig":
        """Parse from a NeMo-style encoder config dict (original checkpoint)."""
        ff_factor = enc.get("ff_expansion_factor", 4)
        return cls(
            hidden_size=enc["d_model"],
            num_hidden_layers=enc["n_layers"],
            num_attention_heads=enc["n_heads"],
            intermediate_size=enc["d_model"] * ff_factor,
            conv_kernel_size=enc.get("conv_kernel_size", 9),
            subsampling_factor=enc.get("subsampling_factor", 8),
            subsampling_conv_channels=enc.get("subsampling_conv_channels", 256),
            num_mel_bins=enc.get("feat_in", 128),
            max_position_embeddings=enc.get("pos_emb_max_len", 5000),
            scale_input=bool(enc.get("xscaling", False)),
            dropout=float(enc.get("dropout", 0.0)),
            dropout_positions=float(enc.get("dropout_emb", 0.0)),
            layerdrop=float(enc.get("layerdrop", 0.0)),
            activation_dropout=float(enc.get("activation_dropout", 0.0)),
            attention_dropout=float(enc.get("dropout_att", 0.0)),
        )

    @classmethod
    def from_mlx_dict(cls, enc: dict[str, Any]) -> "ParakeetEncoderConfig":
        """Parse from an MLX-format encoder config dict (saved by to_dict)."""
        return cls(
            hidden_size=enc["hidden_size"],
            num_hidden_layers=enc["num_hidden_layers"],
            num_attention_heads=enc["num_attention_heads"],
            intermediate_size=enc["intermediate_size"],
            hidden_act=enc.get("hidden_act", "silu"),
            conv_kernel_size=enc.get("conv_kernel_size", 9),
            subsampling_factor=enc.get("subsampling_factor", 8),
            subsampling_conv_channels=enc.get("subsampling_conv_channels", 256),
            num_mel_bins=enc.get("num_mel_bins", 128),
            subsampling_conv_kernel_size=enc.get("subsampling_conv_kernel_size", 3),
            subsampling_conv_stride=enc.get("subsampling_conv_stride", 2),
            max_position_embeddings=enc.get("max_position_embeddings", 5000),
            scale_input=bool(enc.get("scale_input", False)),
            attention_bias=bool(enc.get("attention_bias", True)),
            convolution_bias=bool(enc.get("convolution_bias", True)),
            dropout=float(enc.get("dropout", 0.0)),
            dropout_positions=float(enc.get("dropout_positions", 0.0)),
            layerdrop=float(enc.get("layerdrop", 0.0)),
            activation_dropout=float(enc.get("activation_dropout", 0.0)),
            attention_dropout=float(enc.get("attention_dropout", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "conv_kernel_size": self.conv_kernel_size,
            "subsampling_factor": self.subsampling_factor,
            "subsampling_conv_channels": self.subsampling_conv_channels,
            "num_mel_bins": self.num_mel_bins,
            "subsampling_conv_kernel_size": self.subsampling_conv_kernel_size,
            "subsampling_conv_stride": self.subsampling_conv_stride,
            "max_position_embeddings": self.max_position_embeddings,
            "scale_input": self.scale_input,
            "attention_bias": self.attention_bias,
            "convolution_bias": self.convolution_bias,
            "dropout": self.dropout,
            "dropout_positions": self.dropout_positions,
            "layerdrop": self.layerdrop,
            "activation_dropout": self.activation_dropout,
            "attention_dropout": self.attention_dropout,
        }


@dataclass(frozen=True)
class CohereAsrDecoderConfig:
    """Transformer decoder configuration."""

    vocab_size: int = 16384
    hidden_size: int = 1024
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    intermediate_size: int = 4096
    hidden_act: str = "relu"
    max_position_embeddings: int = 1024
    # encoder output size — used by decoder.proj: Linear(encoder_hidden → hidden)
    encoder_hidden_size: int = 1280
    attention_bias: bool = True
    pad_token_id: int = 2
    eos_token_id: int = 3
    bos_token_id: int = 4
    decoder_start_token_id: int = 13764

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_nemo_dict(
        cls,
        payload: dict[str, Any],
        gen_config: dict[str, Any] | None = None,
    ) -> "CohereAsrDecoderConfig":
        """Parse from a NeMo-style transf_decoder config dict."""
        dec = payload.get("transf_decoder", {}).get("config_dict", {})
        gen = gen_config or {}
        return cls(
            vocab_size=payload.get("vocab_size", 16384),
            hidden_size=dec.get("hidden_size", 1024),
            num_hidden_layers=dec.get("num_layers", 8),
            num_attention_heads=dec.get("num_attention_heads", 8),
            intermediate_size=dec.get("inner_size", 4096),
            hidden_act=dec.get("hidden_act", "relu"),
            max_position_embeddings=dec.get("max_sequence_length", 1024),
            encoder_hidden_size=dec.get("lm_dec_hidden", 1280),
            pad_token_id=gen.get("pad_token_id", 2),
            eos_token_id=gen.get("eos_token_id", 3),
            bos_token_id=gen.get("bos_token_id", 4),
            decoder_start_token_id=gen.get("decoder_start_token_id", 13764),
        )

    @classmethod
    def from_mlx_dict(cls, dec: dict[str, Any]) -> "CohereAsrDecoderConfig":
        """Parse from an MLX-format decoder config dict (saved by to_dict)."""
        return cls(
            vocab_size=dec.get("vocab_size", 16384),
            hidden_size=dec.get("hidden_size", 1024),
            num_hidden_layers=dec.get("num_hidden_layers", 8),
            num_attention_heads=dec.get("num_attention_heads", 8),
            num_key_value_heads=dec.get("num_key_value_heads", 8),
            intermediate_size=dec.get("intermediate_size", 4096),
            hidden_act=dec.get("hidden_act", "relu"),
            max_position_embeddings=dec.get("max_position_embeddings", 1024),
            encoder_hidden_size=dec.get("encoder_hidden_size", 1280),
            attention_bias=bool(dec.get("attention_bias", True)),
            pad_token_id=dec.get("pad_token_id", 2),
            eos_token_id=dec.get("eos_token_id", 3),
            bos_token_id=dec.get("bos_token_id", 4),
            decoder_start_token_id=dec.get("decoder_start_token_id", 13764),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
            "max_position_embeddings": self.max_position_embeddings,
            "encoder_hidden_size": self.encoder_hidden_size,
            "attention_bias": self.attention_bias,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "decoder_start_token_id": self.decoder_start_token_id,
        }


@dataclass(frozen=True)
class CohereAsrConfig:
    """Top-level MLX config for cohere-transcribe-03-2026."""

    encoder: ParakeetEncoderConfig = field(default_factory=ParakeetEncoderConfig)
    decoder: CohereAsrDecoderConfig = field(default_factory=CohereAsrDecoderConfig)
    model_type: str = "cohere_asr"
    sample_rate: int = 16000
    max_audio_clip_s: float = 35.0
    overlap_chunk_s: float = 5.0
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], gen_config: dict[str, Any] | None = None) -> "CohereAsrConfig":
        enc_payload = payload["encoder"]
        # Detect format: NeMo uses "d_model", MLX uses "hidden_size"
        if "d_model" in enc_payload:
            encoder = ParakeetEncoderConfig.from_nemo_dict(enc_payload)
            decoder = CohereAsrDecoderConfig.from_nemo_dict(payload, gen_config=gen_config)
        else:
            encoder = ParakeetEncoderConfig.from_mlx_dict(enc_payload)
            decoder = CohereAsrDecoderConfig.from_mlx_dict(payload["decoder"])
        _consumed = {"encoder", "decoder", "model_type", "sample_rate",
                     "max_audio_clip_s", "overlap_chunk_s", "overlap_chunk_second",
                     "transf_decoder", "vocab_size"}
        extra = {k: v for k, v in payload.items() if k not in _consumed}
        return cls(
            encoder=encoder,
            decoder=decoder,
            model_type=payload.get("model_type", "cohere_asr"),
            sample_rate=payload.get("sample_rate", 16000),
            max_audio_clip_s=float(payload.get("max_audio_clip_s", 35.0)),
            overlap_chunk_s=float(payload.get("overlap_chunk_second", payload.get("overlap_chunk_s", 5.0))),
            extra=extra,
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "CohereAsrConfig":
        model_dir = Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        gen_config: dict[str, Any] | None = None
        gen_path = model_dir / "generation_config.json"
        if gen_path.exists():
            with gen_path.open(encoding="utf-8") as f:
                gen_config = json.load(f)
        return cls.from_dict(payload, gen_config=gen_config)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model_type": self.model_type,
            "sample_rate": self.sample_rate,
            "max_audio_clip_s": self.max_audio_clip_s,
            "overlap_chunk_s": self.overlap_chunk_s,
            "encoder": self.encoder.to_dict(),
            "decoder": self.decoder.to_dict(),
        }
        payload.update(self.extra)
        return payload
