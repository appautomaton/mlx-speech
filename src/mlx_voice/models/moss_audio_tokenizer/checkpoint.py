"""Checkpoint helpers for the Moss audio tokenizer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints import get_openmoss_v0_layouts
from ...checkpoints.sharded import load_state_dict
from .config import MossAudioTokenizerConfig

if TYPE_CHECKING:
    from .model import MossAudioTokenizerModel


@dataclass(frozen=True)
class MossAudioTokenizerCheckpoint:
    """Loaded codec checkpoint after MLX-side sanitization."""

    model_dir: Path
    config: MossAudioTokenizerConfig
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]


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


@dataclass(frozen=True)
class QuantizationConfig:
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
class LoadedMossAudioTokenizerModel:
    model_dir: Path
    config: MossAudioTokenizerConfig
    model: MossAudioTokenizerModel
    checkpoint: MossAudioTokenizerCheckpoint
    alignment_report: AlignmentReport
    quantization: QuantizationConfig | None = None


def prepare_runtime_state_dict(
    state_dict: dict[str, mx.array],
    *,
    quantization: QuantizationConfig | None,
) -> dict[str, mx.array]:
    """Preserve checkpoint dtypes for runtime loading."""

    _ = quantization
    return dict(state_dict)


class SupportsLoadWeights(Protocol):
    def parameters(self): ...

    def load_weights(self, file_or_weights, strict: bool = True): ...


def _restore_weight_norm_weight(weight_g: mx.array, weight_v: mx.array) -> mx.array:
    norm = mx.sqrt(mx.sum(weight_v.astype(mx.float32) ** 2, axis=(1, 2), keepdims=True))
    restored = weight_g.astype(mx.float32) * weight_v.astype(mx.float32) / norm
    return restored.astype(weight_v.dtype)


def get_quantization_config(config: MossAudioTokenizerConfig) -> QuantizationConfig | None:
    payload = config.extra.get("quantization")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("`quantization` must be a dictionary when present in config.json.")
    return QuantizationConfig.from_dict(payload)


def sanitize_state_dict(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Restore weight-normalized 1x1 conv tensors and keep full codec weights."""

    sanitized: dict[str, mx.array] = {}
    skipped_keys: list[str] = []
    renamed_keys: list[tuple[str, str]] = []
    pending_weight_norm: dict[str, dict[str, mx.array]] = {}

    for key, value in weights.items():
        if ".parametrizations.weight.original0" in key:
            base_key = key.replace(".parametrizations.weight.original0", "")
            pending_weight_norm.setdefault(base_key, {})["g"] = value
            renamed_keys.append((key, f"{base_key}.weight"))
            continue
        if ".parametrizations.weight.original1" in key:
            base_key = key.replace(".parametrizations.weight.original1", "")
            pending_weight_norm.setdefault(base_key, {})["v"] = value
            renamed_keys.append((key, f"{base_key}.weight"))
            continue

        if key in sanitized:
            raise ValueError(f"Duplicate tensor key after sanitization: {key}")
        sanitized[key] = value

    for base_key, tensors in pending_weight_norm.items():
        if "g" not in tensors or "v" not in tensors:
            raise ValueError(f"Incomplete weight-norm tensors for {base_key}")
        restored_key = f"{base_key}.weight"
        if restored_key in sanitized:
            raise ValueError(f"Duplicate tensor key after weight restoration: {restored_key}")
        sanitized[restored_key] = _restore_weight_norm_weight(tensors["g"], tensors["v"])

    return sanitized, tuple(sorted(skipped_keys)), tuple(renamed_keys)


def load_moss_audio_tokenizer_checkpoint(
    model_dir: str | Path,
) -> MossAudioTokenizerCheckpoint:
    resolved_dir = Path(model_dir)
    config = MossAudioTokenizerConfig.from_path(resolved_dir)
    loaded = load_state_dict(resolved_dir)
    state_dict, skipped_keys, renamed_keys = sanitize_state_dict(loaded.weights)
    return MossAudioTokenizerCheckpoint(
        model_dir=resolved_dir,
        config=config,
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped_keys,
        renamed_keys=renamed_keys,
    )


def resolve_moss_audio_tokenizer_model_dir(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
) -> Path:
    if model_dir is not None:
        return Path(model_dir)

    layout = get_openmoss_v0_layouts().audio_tokenizer
    quantized_dir = layout.mlx_int8_dir
    original_dir = layout.original_dir
    if prefer_mlx_int8 and any(quantized_dir.glob("*.safetensors")):
        return quantized_dir
    if any(original_dir.glob("*.safetensors")):
        return original_dir
    return quantized_dir if prefer_mlx_int8 else original_dir


def validate_checkpoint_against_model(
    model: SupportsLoadWeights,
    checkpoint: MossAudioTokenizerCheckpoint,
) -> AlignmentReport:
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
    checkpoint: MossAudioTokenizerCheckpoint,
    *,
    strict: bool = True,
) -> AlignmentReport:
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


def _is_quantizable_module(module: Any, *, group_size: int) -> bool:
    return (
        hasattr(module, "weight")
        and hasattr(module, "to_quantized")
        and module.weight.shape[-1] % group_size == 0
    )


def quantize_moss_audio_tokenizer_model(
    model: MossAudioTokenizerModel,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> MossAudioTokenizerModel:
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


def save_moss_audio_tokenizer_model(
    model: MossAudioTokenizerModel,
    model_dir: str | Path,
    *,
    config: MossAudioTokenizerConfig,
    quantization: QuantizationConfig | None = None,
) -> Path:
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


def load_moss_audio_tokenizer_model(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
    strict: bool = True,
) -> LoadedMossAudioTokenizerModel:
    from .model import MossAudioTokenizerModel

    resolved_dir = resolve_moss_audio_tokenizer_model_dir(model_dir, prefer_mlx_int8=prefer_mlx_int8)

    def build_loaded_codec(resolved: Path) -> LoadedMossAudioTokenizerModel:
        checkpoint = load_moss_audio_tokenizer_checkpoint(resolved)
        model = MossAudioTokenizerModel(checkpoint.config)
        quantization = get_quantization_config(checkpoint.config)
        runtime_state_dict = prepare_runtime_state_dict(
            checkpoint.state_dict,
            quantization=quantization,
        )
        if quantization is not None:
            quantize_moss_audio_tokenizer_model(
                model,
                quantization,
                state_dict=runtime_state_dict,
            )
        runtime_checkpoint = MossAudioTokenizerCheckpoint(
            model_dir=checkpoint.model_dir,
            config=checkpoint.config,
            state_dict=runtime_state_dict,
            source_files=checkpoint.source_files,
            skipped_keys=checkpoint.skipped_keys,
            renamed_keys=checkpoint.renamed_keys,
        )
        alignment_report = load_checkpoint_into_model(model, runtime_checkpoint, strict=strict)
        return LoadedMossAudioTokenizerModel(
            model_dir=resolved,
            config=checkpoint.config,
            model=model,
            checkpoint=runtime_checkpoint,
            alignment_report=alignment_report,
            quantization=quantization,
        )

    try:
        return build_loaded_codec(resolved_dir)
    except ValueError:
        if model_dir is not None or not prefer_mlx_int8:
            raise
        fallback_dir = resolve_moss_audio_tokenizer_model_dir(None, prefer_mlx_int8=False)
        if fallback_dir == resolved_dir:
            raise
        return build_loaded_codec(fallback_dir)
