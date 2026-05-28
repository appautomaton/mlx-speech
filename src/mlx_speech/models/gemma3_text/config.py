"""Configuration helpers for the pure-MLX Gemma 3 text-only backbone.

Mirrors the subset of `transformers.Gemma3TextConfig` we actually use. The
goal is to be load-compatible with the converted `gemma_3_12b_it_4bit`
checkpoint directory under `models/`, where the saved `config.json` keeps
the original Gemma 3 text-config payload under the ``text_config`` key plus
a sibling ``quantization`` block.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GemmaRopeScaling:
    """Linear-style RoPE scaling for the global-attention RoPE.

    The local (sliding) RoPE uses its own base frequency
    (``rope_local_base_freq``) and has no scaling applied.
    """

    factor: float = 8.0
    rope_type: str = "linear"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GemmaRopeScaling":
        if payload is None:
            return cls()
        return cls(
            factor=float(payload.get("factor", 8.0)),
            rope_type=str(payload.get("rope_type", "linear")),
        )


@dataclass(frozen=True)
class GemmaTextConfig:
    """Subset of Gemma 3 text-config used by `mlx_speech`."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_local_base_freq: float = 10_000.0
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    query_pre_attn_scalar: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    max_position_embeddings: int = 131_072
    rope_scaling: GemmaRopeScaling = field(default_factory=GemmaRopeScaling)
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def attention_scaling(self) -> float:
        """The scale applied to Q before attention. Pre-attention scalar in
        Gemma 3 is the per-head dim (256), not necessarily ``head_dim``."""
        return self.query_pre_attn_scalar ** -0.5

    def layer_types(self) -> list[str]:
        """Per-layer attention type. Every ``sliding_window_pattern``-th
        layer (1-indexed) is full attention; the rest are sliding."""
        return [
            "sliding_attention" if (i + 1) % self.sliding_window_pattern != 0 else "full_attention"
            for i in range(self.num_hidden_layers)
        ]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GemmaTextConfig":
        # Accept either a flat text-config payload or a wrapper with a
        # ``text_config`` field (the converted checkpoint stores the wrapper).
        if "text_config" in payload and isinstance(payload["text_config"], dict):
            payload = payload["text_config"]

        rope_scaling = GemmaRopeScaling.from_dict(payload.get("rope_scaling"))

        field_map = {
            "hidden_size": int(payload["hidden_size"]),
            "intermediate_size": int(payload["intermediate_size"]),
            "num_hidden_layers": int(payload["num_hidden_layers"]),
            "num_attention_heads": int(payload["num_attention_heads"]),
            "num_key_value_heads": int(payload["num_key_value_heads"]),
            "head_dim": int(payload["head_dim"]),
            "vocab_size": int(payload["vocab_size"]),
            "rms_norm_eps": float(payload.get("rms_norm_eps", 1e-6)),
            "rope_theta": float(payload.get("rope_theta", 1_000_000.0)),
            "rope_local_base_freq": float(payload.get("rope_local_base_freq", 10_000.0)),
            "sliding_window": int(payload.get("sliding_window", 1024)),
            "sliding_window_pattern": int(payload.get("sliding_window_pattern", 6)),
            "query_pre_attn_scalar": int(payload.get("query_pre_attn_scalar", 256)),
            "hidden_activation": str(payload.get("hidden_activation", "gelu_pytorch_tanh")),
            "max_position_embeddings": int(payload.get("max_position_embeddings", 131_072)),
            "rope_scaling": rope_scaling,
        }
        known = set(field_map.keys()) | {"rope_scaling"}
        extra = {k: v for k, v in payload.items() if k not in known}
        return cls(**field_map, extra=extra)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "GemmaTextConfig":
        config_path = Path(model_dir) / "config.json"
        with config_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return cls.from_dict(payload)


__all__ = ["GemmaTextConfig", "GemmaRopeScaling"]
