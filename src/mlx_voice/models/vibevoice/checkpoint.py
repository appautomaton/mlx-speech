"""Checkpoint loading helpers for VibeVoice Large."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints.sharded import load_state_dict
from .config import VibeVoiceConfig

SKIP_KEY_SUFFIXES = (
    "rotary_emb.inv_freq",
    "position_ids",
)


@dataclass(frozen=True)
class VibeVoiceCheckpoint:
    """Loaded VibeVoice checkpoint plus config and loader metadata."""

    model_dir: Path
    config: VibeVoiceConfig
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
    def from_dict(cls, payload: dict[str, Any]) -> QuantizationConfig:
        return cls(
            bits=int(payload["bits"]),
            group_size=int(payload["group_size"]),
            mode=str(payload.get("mode", "affine")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass
class LoadedVibeVoiceModel:
    """A fully materialized VibeVoice MLX model from a local checkpoint."""

    model_dir: Path
    config: VibeVoiceConfig
    model: Any  # VibeVoiceForConditionalGeneration
    checkpoint: VibeVoiceCheckpoint
    alignment_report: AlignmentReport
    quantization: QuantizationConfig | None = None


def _should_skip_key(key: str) -> bool:
    return key.endswith(SKIP_KEY_SUFFIXES)


def _is_conv1d_weight(key: str) -> bool:
    """Check if a checkpoint key is a Conv1d weight (not ConvTranspose1d)."""
    return (
        key.endswith((".conv.conv.weight", ".conv.conv.conv.weight"))
        and "convtr" not in key
    )


def _is_convtr_weight(key: str) -> bool:
    """Check if a checkpoint key is a ConvTranspose1d weight."""
    return key.endswith(".convtr.convtr.weight")


def sanitize_state_dict(
    weights: dict[str, mx.array],
    *,
    is_mlx_native: bool = False,
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Normalize checkpoint keys and weight shapes for MLX loading.

    When ``is_mlx_native`` is False (loading from PyTorch/HF checkpoint):
    - Conv1d weight transpose: PyTorch (out, in/g, k) → MLX (out, k, in/g)
    - ConvTranspose1d weight transpose: PyTorch (in, out, k) → MLX (out, k, in)

    When ``is_mlx_native`` is True (loading from our own MLX-saved checkpoint):
    - No weight transpositions (already in MLX format)
    """

    sanitized: dict[str, mx.array] = {}
    skipped: list[str] = []
    renamed: list[tuple[str, str]] = []

    for key, value in weights.items():
        if _should_skip_key(key):
            skipped.append(key)
            continue

        new_key = key

        if not is_mlx_native:
            # Conv1d: PyTorch (C_out, C_in/groups, K) → MLX (C_out, K, C_in/groups)
            if _is_conv1d_weight(new_key) and value.ndim == 3:
                value = mx.transpose(value, (0, 2, 1))

            # ConvTranspose1d: PyTorch (C_in, C_out, K) → MLX (C_out, K, C_in)
            if _is_convtr_weight(new_key) and value.ndim == 3:
                value = mx.transpose(value, (1, 2, 0))

        if new_key in sanitized:
            raise ValueError(f"Duplicate key after sanitization: {new_key}")
        sanitized[new_key] = value
        if new_key != key:
            renamed.append((key, new_key))

    return sanitized, tuple(skipped), tuple(renamed)


def get_quantization_config(config: VibeVoiceConfig) -> QuantizationConfig | None:
    payload = config.extra.get("quantization")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("`quantization` must be a dict when present in config.json.")
    return QuantizationConfig.from_dict(payload)


def load_vibevoice_checkpoint(model_dir: str | Path) -> VibeVoiceCheckpoint:
    """Load config and sharded safetensors for VibeVoice from a local path."""

    resolved = Path(model_dir)
    config = VibeVoiceConfig.from_path(resolved)
    loaded = load_state_dict(resolved)
    # MLX-native checkpoints (from our converter) have quantization metadata
    is_mlx = config.extra.get("quantization") is not None
    state_dict, skipped, renamed = sanitize_state_dict(
        loaded.weights, is_mlx_native=is_mlx,
    )
    return VibeVoiceCheckpoint(
        model_dir=resolved,
        config=config,
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped,
        renamed_keys=renamed,
    )


def resolve_vibevoice_model_dir(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
) -> Path:
    """Resolve the local VibeVoice checkpoint path."""

    if model_dir is not None:
        return Path(model_dir)

    base = Path("models/vibevoice")
    int8_dir = base / "mlx-int8"
    original_dir = base / "original"

    if prefer_mlx_int8 and any(int8_dir.glob("*.safetensors")):
        return int8_dir
    if any(original_dir.glob("*.safetensors")):
        return original_dir
    return int8_dir if prefer_mlx_int8 else original_dir


def validate_checkpoint_against_model(
    model: SupportsLoadWeights,
    checkpoint: VibeVoiceCheckpoint,
) -> AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    ckpt_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & ckpt_keys):
        m_shape = tuple(int(d) for d in model_params[key].shape)
        c_shape = tuple(int(d) for d in checkpoint.state_dict[key].shape)
        if m_shape != c_shape:
            shape_mismatches.append((key, m_shape, c_shape))

    return AlignmentReport(
        missing_in_model=tuple(sorted(ckpt_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - ckpt_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_checkpoint_into_model(
    model: SupportsLoadWeights,
    checkpoint: VibeVoiceCheckpoint,
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
            lines.append(f"  checkpoint-only (first 10): {report.missing_in_model[:10]}")
        if report.missing_in_checkpoint:
            lines.append(f"  model-only (first 10): {report.missing_in_checkpoint[:10]}")
        if report.shape_mismatches:
            lines.append(f"  shape mismatch (first 5): {report.shape_mismatches[:5]}")
        raise ValueError("\n".join(lines))
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return report


def _is_quantizable_module(module: Any, *, group_size: int) -> bool:
    return (
        hasattr(module, "weight")
        and hasattr(module, "to_quantized")
        and module.weight.shape[-1] % group_size == 0
    )


def quantize_vibevoice_model(
    model: Any,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> Any:
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


def save_vibevoice_model(
    model: Any,
    model_dir: str | Path,
    *,
    config: VibeVoiceConfig,
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


def load_vibevoice_model(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
    strict: bool = True,
) -> LoadedVibeVoiceModel:
    """Load VibeVoice from a local checkpoint directory."""

    from .model import VibeVoiceForConditionalGeneration

    resolved = resolve_vibevoice_model_dir(model_dir, prefer_mlx_int8=prefer_mlx_int8)
    checkpoint = load_vibevoice_checkpoint(resolved)
    model = VibeVoiceForConditionalGeneration(checkpoint.config)
    quantization = get_quantization_config(checkpoint.config)

    if quantization is not None:
        quantize_vibevoice_model(
            model, quantization, state_dict=checkpoint.state_dict,
        )

    alignment = load_checkpoint_into_model(model, checkpoint, strict=strict)
    return LoadedVibeVoiceModel(
        model_dir=resolved,
        config=checkpoint.config,
        model=model,
        checkpoint=checkpoint,
        alignment_report=alignment,
        quantization=quantization,
    )
