"""Checkpoint loading and BF16 packaging for Qwen3-ASR."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

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


def _copy_supporting_files(input_dir: Path, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".safetensors":
            continue
        if path.name == "model.safetensors.index.json":
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
