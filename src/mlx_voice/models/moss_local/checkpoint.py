"""Checkpoint loading helpers for MossTTSLocal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints import get_openmoss_v0_layouts
from ...checkpoints.sharded import LoadedStateDict, load_state_dict
from .config import MossTTSLocalConfig

if TYPE_CHECKING:
    from .model import MossTTSLocalModel


SKIP_KEY_SUFFIXES = (
    "position_ids",
    "rotary_emb.inv_freq",
)


@dataclass(frozen=True)
class MossTTSLocalCheckpoint:
    """Loaded MossTTSLocal checkpoint plus config and loader metadata."""

    model_dir: Path
    config: MossTTSLocalConfig
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]

    @property
    def key_count(self) -> int:
        return len(self.state_dict)


class SupportsLoadWeights(Protocol):
    def parameters(self): ...

    def load_weights(self, file_or_weights, strict: bool = True): ...


@dataclass(frozen=True)
class AlignmentReport:
    """Comparison between a checkpoint state dict and an MLX model tree."""

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


@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization metadata stored alongside converted MLX checkpoints."""

    bits: int
    group_size: int
    mode: str = "affine"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuantizationConfig":
        return cls(
            bits=int(payload["bits"]),
            group_size=int(payload["group_size"]),
            mode=str(payload.get("mode", "affine")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


@dataclass
class LoadedMossTTSLocalModel:
    """A fully materialized MLX model loaded from a local checkpoint directory."""

    model_dir: Path
    config: MossTTSLocalConfig
    model: MossTTSLocalModel
    checkpoint: MossTTSLocalCheckpoint
    alignment_report: AlignmentReport
    quantization: QuantizationConfig | None = None


def prepare_runtime_state_dict(
    state_dict: dict[str, mx.array],
    *,
    quantization: QuantizationConfig | None,
) -> dict[str, mx.array]:
    """Preserve checkpoint dtypes for runtime loading.

    Converted `mlx-int8` checkpoints are the default runtime target. Keep the
    original checkpoint tensor dtypes intact instead of globally widening to
    float32 at load time.
    """

    _ = quantization
    return dict(state_dict)


def should_skip_key(key: str) -> bool:
    """Return `True` for non-parameter tensors we do not load into MLX modules."""

    return key.endswith(SKIP_KEY_SUFFIXES)


def get_quantization_config(config: MossTTSLocalConfig) -> QuantizationConfig | None:
    """Read optional quantization metadata from a model config."""

    payload = config.extra.get("quantization")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("`quantization` must be a dictionary when present in config.json.")
    return QuantizationConfig.from_dict(payload)


def sanitize_state_dict(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Normalize upstream weight keys for MLX-side loading.

    Stage 1 keeps upstream parameter names intact except for skipping a few
    known non-parameter tensors. The rename hook is kept explicit so later
    stages can add MLX-specific key rewrites in one place.
    """

    sanitized: dict[str, mx.array] = {}
    skipped_keys: list[str] = []
    renamed_keys: list[tuple[str, str]] = []

    for key, value in weights.items():
        if should_skip_key(key):
            skipped_keys.append(key)
            continue

        sanitized_key = key
        if sanitized_key in sanitized:
            raise ValueError(f"Duplicate tensor key after sanitization: {sanitized_key}")
        sanitized[sanitized_key] = value
        if sanitized_key != key:
            renamed_keys.append((key, sanitized_key))

    return sanitized, tuple(skipped_keys), tuple(renamed_keys)


def load_moss_tts_local_checkpoint(model_dir: str | Path) -> MossTTSLocalCheckpoint:
    """Load config and sharded safetensors for MossTTSLocal from a local path."""

    resolved_dir = Path(model_dir)
    config = MossTTSLocalConfig.from_path(resolved_dir)
    loaded = load_state_dict(resolved_dir)
    state_dict, skipped_keys, renamed_keys = sanitize_state_dict(loaded.weights)
    return MossTTSLocalCheckpoint(
        model_dir=resolved_dir,
        config=config,
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped_keys,
        renamed_keys=renamed_keys,
    )


def load_moss_tts_local_state_dict(model_dir: str | Path) -> LoadedStateDict:
    """Expose the generic loader for scripts that want pre-sanitized weights."""

    return load_state_dict(model_dir)


def resolve_moss_tts_local_model_dir(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
) -> Path:
    """Resolve the default local MossTTSLocal path for runtime loading."""

    if model_dir is not None:
        return Path(model_dir)

    layout = get_openmoss_v0_layouts().moss_tts_local
    quantized_dir = layout.mlx_int8_dir
    original_dir = layout.original_dir

    if prefer_mlx_int8 and any(quantized_dir.glob("*.safetensors")):
        return quantized_dir
    if any(original_dir.glob("*.safetensors")):
        return original_dir
    return quantized_dir if prefer_mlx_int8 else original_dir


def _is_quantizable_module(
    module: Any,
    *,
    group_size: int,
) -> bool:
    return (
        hasattr(module, "weight")
        and hasattr(module, "to_quantized")
        and module.weight.shape[-1] % group_size == 0
    )


def quantize_moss_tts_local_model(
    model: MossTTSLocalModel,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> MossTTSLocalModel:
    """Quantize MossTTSLocal in-place using the MLX module tree."""

    quantized_keys = set(state_dict) if state_dict is not None else None

    def should_quantize(path: str, module: Any) -> bool:
        if not _is_quantizable_module(module, group_size=quantization.group_size):
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


def save_moss_tts_local_model(
    model: MossTTSLocalModel,
    model_dir: str | Path,
    *,
    config: MossTTSLocalConfig,
    quantization: QuantizationConfig | None = None,
) -> Path:
    """Save an MLX-native MossTTSLocal checkpoint directory."""

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = tree_flatten(model.parameters(), destination={})
    mx.eval(list(weights.values()))
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        weights,
        metadata={"format": "mlx"},
    )

    config_payload = config.to_dict()
    if quantization is not None:
        config_payload["quantization"] = quantization.to_dict()
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return output_dir


def validate_checkpoint_against_model(
    model: SupportsLoadWeights,
    checkpoint: MossTTSLocalCheckpoint,
) -> AlignmentReport:
    """Compare checkpoint keys and tensor shapes against an MLX model."""

    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        checkpoint_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return AlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_checkpoint_into_model(
    model: SupportsLoadWeights,
    checkpoint: MossTTSLocalCheckpoint,
    *,
    strict: bool = True,
) -> AlignmentReport:
    """Validate and then load a MossTTSLocal checkpoint into an MLX model."""

    report = validate_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            "Checkpoint/model alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only keys, "
            f"{len(report.missing_in_checkpoint)} model-only keys, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return report


def load_moss_tts_local_model(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
    strict: bool = True,
) -> LoadedMossTTSLocalModel:
    """Load MossTTSLocal from a local checkpoint directory."""

    from .model import MossTTSLocalModel

    resolved_dir = resolve_moss_tts_local_model_dir(
        model_dir,
        prefer_mlx_int8=prefer_mlx_int8,
    )
    checkpoint = load_moss_tts_local_checkpoint(resolved_dir)
    model = MossTTSLocalModel(checkpoint.config)
    quantization = get_quantization_config(checkpoint.config)
    runtime_state_dict = prepare_runtime_state_dict(
        checkpoint.state_dict,
        quantization=quantization,
    )
    if quantization is not None:
        quantize_moss_tts_local_model(
            model,
            quantization,
            state_dict=runtime_state_dict,
        )
    runtime_checkpoint = MossTTSLocalCheckpoint(
        model_dir=checkpoint.model_dir,
        config=checkpoint.config,
        state_dict=runtime_state_dict,
        source_files=checkpoint.source_files,
        skipped_keys=checkpoint.skipped_keys,
        renamed_keys=checkpoint.renamed_keys,
    )
    alignment_report = load_checkpoint_into_model(model, runtime_checkpoint, strict=strict)
    return LoadedMossTTSLocalModel(
        model_dir=resolved_dir,
        config=checkpoint.config,
        model=model,
        checkpoint=runtime_checkpoint,
        alignment_report=alignment_report,
        quantization=quantization,
    )
