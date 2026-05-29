"""Checkpoint loading for Granite Speech ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints.sharded import load_state_dict
from .config import GraniteSpeechConfig


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
