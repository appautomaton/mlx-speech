"""Runtime configuration for Breeze TTS 2.

The public inference path decodes with the bundled audio tokenizer. A Mimi
block may still be present in the original config and is not the codec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

CODEC_NAME = "audio_tokenizer"


@dataclass(frozen=True)
class BreezeTextEncoderConfig:
    """T5Gemma2 text encoder settings used by the public Breeze path."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    query_pre_attn_scalar: int
    sliding_window: int
    eoi_token_index: int
    layer_types: tuple[str, ...]
    full_rope_theta: float
    full_rope_factor: float
    sliding_rope_theta: float
    attention_bias: bool
    backbone_hidden_size: int

    @classmethod
    def from_runtime(cls, payload: Mapping[str, Any]) -> "BreezeTextEncoderConfig":
        marker = payload.get("mlx_speech")
        codec = marker.get("codec") if isinstance(marker, Mapping) else None
        if codec != CODEC_NAME:
            raise ValueError(
                "Breeze runtime config must select the audio_tokenizer codec, "
                f"not {codec!r}."
            )
        text = payload.get("text_encoder_config")
        if not isinstance(text, Mapping):
            raise ValueError("Breeze runtime config is missing text_encoder_config.")
        backbone_hidden_size = payload.get("hidden_size")
        if not isinstance(backbone_hidden_size, int):
            raise ValueError("Breeze runtime config is missing hidden_size.")
        return cls._from_text(text, backbone_hidden_size=backbone_hidden_size)

    @classmethod
    def _from_text(
        cls, text: Mapping[str, Any], *, backbone_hidden_size: int
    ) -> "BreezeTextEncoderConfig":
        activation = text.get("hidden_activation")
        if activation != "gelu_pytorch_tanh":
            raise ValueError(f"Unsupported text encoder activation {activation!r}.")
        if text.get("attn_logit_softcapping") is not None:
            raise ValueError("Text attention soft-capping is not implemented.")
        layer_types = tuple(str(item) for item in text["layer_types"])
        unknown = sorted(set(layer_types) - {"full_attention", "sliding_attention"})
        if unknown:
            raise ValueError(f"Unsupported text attention types: {unknown}.")
        num_layers = int(text["num_hidden_layers"])
        if len(layer_types) != num_layers:
            raise ValueError(
                "Text encoder layer_types do not match num_hidden_layers: "
                f"{len(layer_types)} vs {num_layers}."
            )
        heads = int(text["num_attention_heads"])
        kv_heads = int(text["num_key_value_heads"])
        if kv_heads <= 0 or heads % kv_heads != 0:
            raise ValueError(
                f"Text attention heads {heads} are not divisible by {kv_heads}."
            )
        ropes = text["rope_parameters"]
        full = ropes["full_attention"]
        sliding = ropes["sliding_attention"]
        if full.get("rope_type") != "linear":
            raise ValueError(
                f"Unsupported full-attention RoPE {full.get('rope_type')!r}."
            )
        if sliding.get("rope_type", "default") != "default":
            raise ValueError(
                f"Unsupported sliding-attention RoPE {sliding.get('rope_type')!r}."
            )
        sliding_window = int(text["sliding_window"])
        if sliding_window <= 0:
            raise ValueError("Text sliding_window must be positive.")
        return cls(
            vocab_size=int(text["vocab_size"]),
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            num_hidden_layers=num_layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=int(text["head_dim"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            query_pre_attn_scalar=int(text["query_pre_attn_scalar"]),
            sliding_window=sliding_window,
            eoi_token_index=int(text["eoi_token_index"]),
            layer_types=layer_types,
            full_rope_theta=float(full["rope_theta"]),
            full_rope_factor=float(full["factor"]),
            sliding_rope_theta=float(sliding["rope_theta"]),
            attention_bias=bool(text.get("attention_bias", False)),
            backbone_hidden_size=backbone_hidden_size,
        )


def load_text_encoder_config(model_dir: Path) -> BreezeTextEncoderConfig:
    """Read the text encoder settings from a converted runtime package."""

    path = model_dir.expanduser().resolve() / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return BreezeTextEncoderConfig.from_runtime(payload)


@dataclass(frozen=True)
class RopeScaling:
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int


@dataclass(frozen=True)
class BreezeDecoderConfig:
    """One causal stack: the Qwen3 backbone or the depth decoder."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    qk_norm: bool
    rope_scaling: RopeScaling | None = None
    cache_growth: int = 256

    def __post_init__(self) -> None:
        if (
            self.num_key_value_heads <= 0
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError(
                f"Attention heads {self.num_attention_heads} are not divisible by "
                f"{self.num_key_value_heads}."
            )
        if (
            min(
                self.hidden_size,
                self.head_dim,
                self.num_hidden_layers,
                self.max_position_embeddings,
            )
            <= 0
        ):
            raise ValueError("Decoder dimensions must be positive.")


@dataclass(frozen=True)
class BreezeRuntimeConfig:
    text_encoder: BreezeTextEncoderConfig
    backbone: BreezeDecoderConfig
    depth: BreezeDecoderConfig
    vocab_size: int
    num_codebooks: int
    audio_embed_size: int
    text_vocab_size: int

    @property
    def audio_embedding_rows(self) -> int:
        return self.num_codebooks * self.vocab_size

    @property
    def backbone_eos_id(self) -> int:
        return self.vocab_size

    @classmethod
    def from_runtime(cls, payload: Mapping[str, Any]) -> "BreezeRuntimeConfig":
        text = BreezeTextEncoderConfig.from_runtime(payload)
        backbone_payload = payload.get("backbone_config")
        depth_payload = payload.get("depth_decoder_config")
        if not isinstance(backbone_payload, Mapping) or not isinstance(
            depth_payload, Mapping
        ):
            raise ValueError("Breeze runtime config is missing a decoder config.")
        if backbone_payload.get("rope_scaling") is not None:
            raise ValueError("The Qwen3 backbone config is not expected to scale RoPE.")
        if payload.get("backbone_model_type") != "qwen3":
            raise ValueError(
                f"Unsupported backbone {payload.get('backbone_model_type')!r}; expected qwen3."
            )
        vocab_size = int(payload["vocab_size"])
        num_codebooks = int(payload["num_codebooks"])
        audio_embed_size = int(payload["audio_embed_size"])
        if int(payload["hidden_size"]) != audio_embed_size:
            raise ValueError(
                "Audio embeddings require a projector, which this package does not use."
            )
        if int(depth_payload["backbone_hidden_size"]) != audio_embed_size:
            raise ValueError("Depth decoder backbone projector is not implemented.")
        if int(depth_payload["audio_embed_size"]) != audio_embed_size:
            raise ValueError(
                "Depth decoder audio embedding width does not match the runtime config."
            )
        return cls(
            text_encoder=text,
            backbone=_decoder_config(backbone_payload, qk_norm=True, cache_growth=256),
            depth=_decoder_config(depth_payload, qk_norm=False, cache_growth=16),
            vocab_size=vocab_size,
            num_codebooks=num_codebooks,
            audio_embed_size=audio_embed_size,
            text_vocab_size=int(payload["text_vocab_size"]),
        )


def _decoder_config(
    payload: Mapping[str, Any], *, qk_norm: bool, cache_growth: int
) -> BreezeDecoderConfig:
    if payload.get("hidden_act", "silu") != "silu":
        raise ValueError(
            f"Unsupported decoder activation {payload.get('hidden_act')!r}."
        )
    if payload.get("attention_bias", False) or payload.get("mlp_bias", False):
        raise ValueError("Decoder projection bias is not implemented.")
    scaling = payload.get("rope_scaling")
    rope_scaling = None
    if scaling is not None:
        if not isinstance(scaling, Mapping) or scaling.get("rope_type") != "llama3":
            raise ValueError(f"Unsupported decoder RoPE scaling {scaling!r}.")
        rope_scaling = RopeScaling(
            factor=float(scaling["factor"]),
            low_freq_factor=float(scaling["low_freq_factor"]),
            high_freq_factor=float(scaling["high_freq_factor"]),
            original_max_position_embeddings=int(
                scaling["original_max_position_embeddings"]
            ),
        )
    return BreezeDecoderConfig(
        hidden_size=int(payload["hidden_size"]),
        intermediate_size=int(payload["intermediate_size"]),
        num_hidden_layers=int(payload["num_hidden_layers"]),
        num_attention_heads=int(payload["num_attention_heads"]),
        num_key_value_heads=int(payload["num_key_value_heads"]),
        head_dim=int(payload["head_dim"]),
        rms_norm_eps=float(payload["rms_norm_eps"]),
        rope_theta=float(payload["rope_theta"]),
        max_position_embeddings=int(payload["max_position_embeddings"]),
        qk_norm=qk_norm,
        rope_scaling=rope_scaling,
        cache_growth=cache_growth,
    )


def load_runtime_config(model_dir: Path) -> BreezeRuntimeConfig:
    """Read decoder settings from a converted runtime package."""

    path = model_dir.expanduser().resolve() / "config.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return BreezeRuntimeConfig.from_runtime(payload)
