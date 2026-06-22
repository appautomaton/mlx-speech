"""Checkpoint loading, quantization, and packaging for Qwen3-ASR."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints.sharded import load_state_dict
from .config import Qwen3ASRConfig


GENERATED_MODEL_ONLY_KEYS = frozenset(
    {
        "audio_tower.positional_embedding.positional_embedding",
    }
)


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int = 8
    group_size: int = 64
    mode: str = "affine"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuantizationConfig":
        return cls(
            bits=int(d["bits"]),
            group_size=int(d["group_size"]),
            mode=str(d.get("mode", "affine")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True)
class AlignmentReport:
    checkpoint_only: tuple[str, ...]
    model_only: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return not self.checkpoint_only and not self.model_only and not self.shape_mismatches

    @property
    def unexpected_model_only(self) -> tuple[str, ...]:
        return tuple(key for key in self.model_only if key not in GENERATED_MODEL_ONLY_KEYS)

    @property
    def is_loadable_match(self) -> bool:
        return (
            not self.checkpoint_only
            and not self.unexpected_model_only
            and not self.shape_mismatches
        )


@dataclass(frozen=True)
class Qwen3ASRCheckpoint:
    model_dir: Path
    config: Qwen3ASRConfig
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]
    transposed_keys: tuple[str, ...]


@dataclass(frozen=True)
class Qwen3ASRConversionReport:
    input_dir: Path
    output_dir: Path
    tensor_count: int
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]
    transposed_keys: tuple[str, ...]
    copied_files: tuple[Path, ...]
    output_file: Path


def sanitize_key(key: str) -> str | None:
    if key.startswith("thinker.audio_tower."):
        return "audio_tower." + key.removeprefix("thinker.audio_tower.")
    if key.startswith("thinker.model."):
        return "text_decoder.model." + key.removeprefix("thinker.model.")
    if key.startswith("thinker.lm_head."):
        return "text_decoder.lm_head." + key.removeprefix("thinker.lm_head.")

    if key.startswith("audio_tower.") or key.startswith("text_decoder."):
        return key
    return None


def sanitize_state_dict(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...], tuple[str, ...]]:
    sanitized: dict[str, mx.array] = {}
    skipped: list[str] = []
    renamed: list[tuple[str, str]] = []
    transposed: list[str] = []

    for original_key, value in weights.items():
        key = sanitize_key(original_key)
        if key is None:
            skipped.append(original_key)
            continue
        if key != original_key:
            renamed.append((original_key, key))

        sanitized_value, was_transposed = _maybe_transpose_audio_conv2d(key, value)
        if key in sanitized:
            raise ValueError(f"Duplicate key after Qwen3-ASR sanitization: {key!r}")
        sanitized[key] = sanitized_value
        if was_transposed:
            transposed.append(key)

    return sanitized, tuple(skipped), tuple(renamed), tuple(transposed)


def load_qwen3_asr_checkpoint(model_dir: str | Path) -> Qwen3ASRCheckpoint:
    model_dir = Path(model_dir)
    loaded = load_state_dict(model_dir)
    state_dict, skipped, renamed, transposed = sanitize_state_dict(loaded.weights)
    return Qwen3ASRCheckpoint(
        model_dir=model_dir,
        config=Qwen3ASRConfig.from_dir(model_dir),
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped,
        renamed_keys=renamed,
        transposed_keys=transposed,
    )


def build_alignment_report(
    model_parameters: dict[str, mx.array],
    checkpoint_state: dict[str, mx.array],
) -> AlignmentReport:
    model_keys = set(model_parameters)
    checkpoint_keys = set(checkpoint_state)
    mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(d) for d in model_parameters[key].shape)
        checkpoint_shape = tuple(int(d) for d in checkpoint_state[key].shape)
        if model_shape != checkpoint_shape:
            mismatches.append((key, model_shape, checkpoint_shape))

    return AlignmentReport(
        checkpoint_only=tuple(sorted(checkpoint_keys - model_keys)),
        model_only=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(mismatches),
    )


def validate_checkpoint_against_model(
    model: nn.Module,
    checkpoint: Qwen3ASRCheckpoint,
) -> AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    return build_alignment_report(model_params, checkpoint.state_dict)


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint: Qwen3ASRCheckpoint,
    *,
    strict: bool = True,
) -> AlignmentReport:
    report = validate_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_loadable_match:
        raise ValueError(
            f"Qwen3-ASR checkpoint alignment failed: "
            f"{len(report.checkpoint_only)} checkpoint-only, "
            f"{len(report.unexpected_model_only)} unexpected model-only, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    effective_strict = strict and not report.model_only
    model.load_weights(list(checkpoint.state_dict.items()), strict=effective_strict)
    return report


def save_qwen3_asr_bf16_checkpoint(
    checkpoint: Qwen3ASRCheckpoint,
    output_dir: str | Path,
    *,
    copy_supporting_files: bool = True,
) -> Qwen3ASRConversionReport:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model.safetensors"
    mx.save_safetensors(str(output_file), checkpoint.state_dict, metadata={"format": "mlx"})

    copied_files: list[Path] = []
    if copy_supporting_files:
        copied_files = _copy_supporting_files(checkpoint.model_dir, output_dir)

    return Qwen3ASRConversionReport(
        input_dir=checkpoint.model_dir,
        output_dir=output_dir,
        tensor_count=len(checkpoint.state_dict),
        skipped_keys=checkpoint.skipped_keys,
        renamed_keys=checkpoint.renamed_keys,
        transposed_keys=checkpoint.transposed_keys,
        copied_files=tuple(copied_files),
        output_file=output_file,
    )


def quantize_qwen3_asr_model(
    model: nn.Module,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> nn.Module:
    """Quantize eligible Linear/Embedding layers in place.

    When ``state_dict`` is provided (the load path), only layers that were
    quantized in the saved checkpoint are re-quantized — matching the exact set
    via their ``.scales`` keys. When it is ``None`` (the convert path), every
    eligible layer is quantized.
    """
    quantized_keys = set(state_dict) if state_dict is not None else None

    def should_quantize(path: str, module: Any) -> bool:
        if not (hasattr(module, "weight") and hasattr(module, "to_quantized")):
            return False
        if module.weight.shape[-1] % quantization.group_size != 0:
            return False
        if quantized_keys is None:
            return True
        return f"{path}.scales" in quantized_keys

    nn.quantize(
        model,
        group_size=quantization.group_size,
        bits=quantization.bits,
        mode=quantization.mode,
        class_predicate=should_quantize,
    )
    return model


def save_qwen3_asr_model(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: Qwen3ASRConfig,
    quantization: QuantizationConfig | None = None,
    copy_supporting_files_from: str | Path | None = None,
) -> Path:
    """Save a (quantized) Qwen3-ASR model as an MLX runtime package.

    Writes ``model.safetensors`` from the live module parameters and a
    ``config.json`` carrying the architecture plus an optional ``quantization``
    block read back on load by :func:`get_quantization_config`.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = tree_flatten(model.parameters(), destination={})
    mx.eval(list(weights.values()))
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        weights,
        metadata={"format": "mlx"},
    )

    # Copy tokenizer/assets first; the quantization-aware config.json is written
    # last so it stays authoritative over any copied original config.json.
    if copy_supporting_files_from is not None:
        _copy_supporting_files(Path(copy_supporting_files_from), output_dir)

    payload = config.to_dict()
    if quantization is not None:
        payload["quantization"] = quantization.to_dict()
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return output_dir


def get_quantization_config(config: Qwen3ASRConfig) -> QuantizationConfig | None:
    q = config.extra.get("quantization")
    if q is None:
        return None
    return QuantizationConfig.from_dict(q)


def _copy_supporting_files(input_dir: Path, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        # Don't carry the upstream README into our package; the published repo's
        # README.md is our own model card (scripts/hugging_face/model_cards/).
        if path.name in {"README.md", ".gitattributes"}:
            continue
        destination = output_dir / path.name
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def _is_audio_conv2d_weight(key: str) -> bool:
    return key in {
        "audio_tower.conv2d1.weight",
        "audio_tower.conv2d2.weight",
        "audio_tower.conv2d3.weight",
    }


def _maybe_transpose_audio_conv2d(key: str, value: mx.array) -> tuple[mx.array, bool]:
    if not _is_audio_conv2d_weight(key) or value.ndim != 4:
        return value, False
    already_mlx_layout = int(value.shape[1]) == 3 and int(value.shape[2]) == 3
    if already_mlx_layout:
        return value, False
    return value.transpose(0, 2, 3, 1), True
