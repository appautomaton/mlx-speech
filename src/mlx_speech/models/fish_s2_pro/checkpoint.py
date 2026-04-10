from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
from mlx.utils import tree_flatten

from ...checkpoints.sharded import load_state_dict
from .config import FishS2ProConfig


@dataclass(frozen=True)
class FishS2ProCheckpoint:
    """Loaded Fish S2 Pro checkpoint plus loader metadata."""

    model_dir: Path
    state_dict: dict[str, mx.array]
    config: FishS2ProConfig
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]


class SupportsLoadWeights(Protocol):
    def parameters(self): ...

    def load_weights(self, file_or_weights, strict: bool = True): ...


@dataclass(frozen=True)
class AlignmentReport:
    missing_in_model: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return (
            not self.missing_in_model
            and not self.missing_in_checkpoint
            and not self.shape_mismatches
        )


def _rename_key(key: str) -> str:
    if key.startswith("text_model.model."):
        return key[len("text_model.model.") :]
    if key.startswith("audio_decoder.codebook_embeddings."):
        return "codebook_embeddings." + key[len("audio_decoder.codebook_embeddings.") :]
    if key.startswith("audio_decoder."):
        return "fast_" + key[len("audio_decoder.") :]
    return key


def sanitize_state_dict(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...]]:
    sanitized: dict[str, mx.array] = {}
    skipped: list[str] = []
    renamed: list[tuple[str, str]] = []

    for key, value in weights.items():
        new_key = _rename_key(key)
        if new_key in sanitized:
            raise ValueError(f"Duplicate key after sanitization: {new_key}")
        sanitized[new_key] = value
        if new_key != key:
            renamed.append((key, new_key))

    return sanitized, tuple(skipped), tuple(renamed)


def load_fish_s2_pro_checkpoint(
    model_dir: str | Path,
) -> FishS2ProCheckpoint:
    """Load Fish S2 Pro checkpoint from a local directory."""

    resolved = Path(model_dir)
    config = FishS2ProConfig.from_path(resolved)
    loaded = load_state_dict(resolved)
    state_dict, skipped, renamed = sanitize_state_dict(loaded.weights)

    return FishS2ProCheckpoint(
        model_dir=resolved,
        state_dict=state_dict,
        config=config,
        source_files=loaded.files,
        skipped_keys=skipped,
        renamed_keys=renamed,
    )


def validate_checkpoint_against_model(
    model: SupportsLoadWeights,
    checkpoint: FishS2ProCheckpoint,
) -> AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    ckpt_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & ckpt_keys):
        model_shape = tuple(int(d) for d in model_params[key].shape)
        checkpoint_shape = tuple(int(d) for d in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return AlignmentReport(
        missing_in_model=tuple(sorted(ckpt_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - ckpt_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_checkpoint_into_model(
    model: SupportsLoadWeights,
    checkpoint: FishS2ProCheckpoint,
    *,
    strict: bool = True,
) -> AlignmentReport:
    report = validate_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        lines = [
            "Checkpoint/model alignment failed:",
            f"  {len(report.missing_in_model)} checkpoint-only keys",
            f"  {len(report.missing_in_checkpoint)} model-only keys",
            f"  {len(report.shape_mismatches)} shape mismatches",
        ]
        if report.missing_in_model:
            lines.append(
                f"  checkpoint-only (first 10): {report.missing_in_model[:10]}"
            )
        if report.missing_in_checkpoint:
            lines.append(
                f"  model-only (first 10): {report.missing_in_checkpoint[:10]}"
            )
        if report.shape_mismatches:
            lines.append(f"  shape mismatch (first 5): {report.shape_mismatches[:5]}")
        raise ValueError("\n".join(lines))

    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return report
