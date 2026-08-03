"""Checkpoint loading for Granite Speech ASR."""

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
from .config import GraniteSpeechConfig


@dataclass(frozen=True)
class QuantizationConfig:
    """MLX weight-only quantization metadata stored with runtime artifacts."""

    bits: int = 8
    group_size: int = 64
    mode: str = "affine"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuantizationConfig":
        return cls(
            bits=int(payload["bits"]),
            group_size=int(payload["group_size"]),
            mode=str(payload.get("mode", "affine")),
        )

    def to_dict(self) -> dict[str, int | str]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class AlignmentReport:
    """Differences between model parameters and checkpoint tensors."""

    checkpoint_only: tuple[str, ...]
    model_only: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return not self.checkpoint_only and not self.model_only and not self.shape_mismatches


@dataclass(frozen=True)
class GraniteSpeechCheckpoint:
    """Loaded Granite Speech checkpoint plus sanitizer provenance."""

    model_dir: Path
    config: GraniteSpeechConfig
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    transposed_keys: tuple[str, ...]


def _is_num_batches_tracked(key: str) -> bool:
    return key.endswith(".num_batches_tracked")


def _is_up_or_down_conv_weight(key: str) -> bool:
    return key.endswith(".conv.up_conv.weight") or key.endswith(".conv.down_conv.weight")


def _is_depth_conv_weight(key: str) -> bool:
    return key.endswith(".conv.depth_conv.conv.weight")


def _maybe_transpose_original_conv1d(key: str, value: mx.array) -> tuple[mx.array, bool]:
    """Transpose original PyTorch Conv1d weights into MLX Conv1d layout."""
    if value.ndim != 3:
        return value, False

    if _is_up_or_down_conv_weight(key):
        already_mlx_layout = value.shape[1] == 1 and value.shape[2] != 1
        if already_mlx_layout:
            return value, False
        return value.transpose(0, 2, 1), True

    if _is_depth_conv_weight(key):
        already_mlx_layout = value.shape[2] == 1 and value.shape[1] != 1
        if already_mlx_layout:
            return value, False
        return value.transpose(0, 2, 1), True

    return value, False


def sanitize_state_dict(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[str, ...]]:
    """Sanitize original Granite Speech checkpoint keys and Conv1d layouts."""
    sanitized: dict[str, mx.array] = {}
    skipped: list[str] = []
    transposed: list[str] = []

    for key, value in weights.items():
        if _is_num_batches_tracked(key):
            skipped.append(key)
            continue

        sanitized_value, was_transposed = _maybe_transpose_original_conv1d(key, value)
        if key in sanitized:
            raise ValueError(f"Duplicate key after sanitization: {key!r}")
        sanitized[key] = sanitized_value
        if was_transposed:
            transposed.append(key)

    return sanitized, tuple(skipped), tuple(transposed)


def load_granite_speech_checkpoint(model_dir: str | Path) -> GraniteSpeechCheckpoint:
    model_dir = Path(model_dir)
    loaded = load_state_dict(model_dir)
    state_dict, skipped, transposed = sanitize_state_dict(loaded.weights)
    return GraniteSpeechCheckpoint(
        model_dir=model_dir,
        config=GraniteSpeechConfig.from_path(model_dir),
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped,
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
    checkpoint: GraniteSpeechCheckpoint,
) -> AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    return build_alignment_report(model_params, checkpoint.state_dict)


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint: GraniteSpeechCheckpoint,
    *,
    strict: bool = True,
) -> AlignmentReport:
    """Load sanitized Granite Speech weights after explicit key/shape accounting."""
    report = validate_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            f"Checkpoint alignment failed: "
            f"{len(report.checkpoint_only)} checkpoint-only, "
            f"{len(report.model_only)} model-only, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return report


def get_quantization_config(
    config: GraniteSpeechConfig,
) -> QuantizationConfig | None:
    """Read the canonical quantization block, accepting the MLX community alias."""
    canonical = config.extra.get("quantization")
    compatibility = config.extra.get("quantization_config")
    if canonical is None and compatibility is None:
        return None
    if canonical is not None and compatibility is not None and canonical != compatibility:
        raise ValueError(
            "Granite Speech quantization and quantization_config blocks disagree"
        )
    payload = canonical if canonical is not None else compatibility
    if not isinstance(payload, dict):
        raise ValueError("Granite Speech quantization metadata must be an object")
    return QuantizationConfig.from_dict(payload)


def quantize_granite_speech_model(
    model: nn.Module,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> nn.Module:
    """Quantize Granite LM Linear/Embedding weights or recreate a saved layout."""
    checkpoint_keys = set(state_dict) if state_dict is not None else None

    def should_quantize(path: str, module: Any) -> bool:
        if not path.startswith("language_model."):
            return False
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if module.weight.shape[-1] % quantization.group_size != 0:
            return False
        if checkpoint_keys is None:
            return True
        return f"{path}.scales" in checkpoint_keys

    nn.quantize(
        model,
        group_size=quantization.group_size,
        bits=quantization.bits,
        mode=quantization.mode,
        class_predicate=should_quantize,
    )
    return model


def save_granite_speech_model(
    model: nn.Module,
    output_dir: str | Path,
    *,
    config: GraniteSpeechConfig,
    quantization: QuantizationConfig | None = None,
    copy_supporting_files_from: str | Path | None = None,
) -> Path:
    """Write a self-contained MLX Granite Speech runtime artifact."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = tree_flatten(model.parameters(), destination={})
    mx.eval(list(weights.values()))
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        weights,
        metadata={"format": "mlx"},
    )

    if copy_supporting_files_from is not None:
        _copy_supporting_files(Path(copy_supporting_files_from), output_dir)

    payload = config.to_dict()
    payload.pop("quantization", None)
    payload.pop("quantization_config", None)
    if quantization is not None:
        payload["quantization"] = quantization.to_dict()
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    return output_dir


def _copy_supporting_files(input_dir: Path, output_dir: Path) -> tuple[Path, ...]:
    copied: list[Path] = []
    for source in sorted(input_dir.iterdir()):
        if not source.is_file():
            continue
        if source.suffix == ".safetensors":
            continue
        if source.name in {
            "model.safetensors.index.json",
            "README.md",
            ".gitattributes",
        }:
            continue
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        copied.append(destination)
    return tuple(copied)
