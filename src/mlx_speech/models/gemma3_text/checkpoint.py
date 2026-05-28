"""Checkpoint loading for the pure-MLX Gemma 3 text backbone.

Reads the converted MLX 4-bit affine checkpoint produced by
`scripts/convert/gemma_3_text_4bit.py`. Steps:

1. Build the model with a config parsed from `config.json`.
2. In-place quantize every `nn.Linear` whose corresponding `.scales` key is
   present in the saved state dict (matches the conversion convention).
3. Concatenate weight shards and call `model.load_weights(...)`.

The quantization metadata comes from the `quantization` block in
`config.json` (group_size=64, bits=4, mode="affine").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .config import GemmaTextConfig
from .model import Gemma3Model


# --------------------------------------------------------------------------- #
# Quantization predicate
# --------------------------------------------------------------------------- #

def _is_quantizable(module: nn.Module, group_size: int) -> bool:
    return (
        isinstance(module, (nn.Linear, nn.Embedding))
        and hasattr(module, "to_quantized")
        and module.weight.shape[-1] % group_size == 0
    )


def _quantize_to_match(
    model: nn.Module,
    state_dict: dict[str, mx.array],
    *,
    bits: int,
    group_size: int,
    mode: str = "affine",
) -> None:
    """In-place quantize modules whose `.scales` key exists in state_dict."""
    quantized_keys = set(state_dict)

    def predicate(path: str, module: Any) -> bool:
        if not _is_quantizable(module, group_size=group_size):
            return False
        return f"{path}.scales" in quantized_keys

    nn.quantize(model, group_size=group_size, bits=bits, mode=mode, class_predicate=predicate)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GemmaQuantization:
    bits: int = 4
    group_size: int = 64
    mode: str = "affine"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GemmaQuantization | None":
        if not payload:
            return None
        return cls(
            bits=int(payload.get("bits", 4)),
            group_size=int(payload.get("group_size", 64)),
            mode=str(payload.get("mode", "affine")),
        )


def _load_shards(model_dir: Path) -> dict[str, mx.array]:
    """Load every `model-*-of-*.safetensors` shard, falling back to a single
    `model.safetensors` if no shard files are present."""
    shards = sorted(model_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        single = model_dir / "model.safetensors"
        if not single.is_file():
            raise FileNotFoundError(f"No safetensors files under {model_dir}")
        shards = [single]
    weights: dict[str, mx.array] = {}
    for shard in shards:
        weights.update(mx.load(str(shard)))
    return weights


def load_gemma3_text_model(
    model_dir: str | Path,
    *,
    strict: bool = True,
) -> tuple[Gemma3Model, GemmaTextConfig]:
    """Construct and load a Gemma 3 text backbone from a local directory.

    Returns ``(model, config)``. The model has its quantized linear/embedding
    layers swapped in and the weights loaded; no further setup is needed
    before calling ``model(input_ids, attention_mask)``.
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    config_path = model_dir / "config.json"
    with config_path.open("r", encoding="utf-8") as fh:
        config_payload = json.load(fh)

    config = GemmaTextConfig.from_dict(config_payload)
    quantization = GemmaQuantization.from_dict(config_payload.get("quantization"))

    model = Gemma3Model(config)

    state_dict = _load_shards(model_dir)

    if quantization is not None:
        _quantize_to_match(
            model,
            state_dict,
            bits=quantization.bits,
            group_size=quantization.group_size,
            mode=quantization.mode,
        )

    model.load_weights(list(state_dict.items()), strict=strict)
    return model, config


__all__ = ["load_gemma3_text_model", "GemmaQuantization"]
