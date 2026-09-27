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
