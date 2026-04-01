"""Checkpoint loading helpers for Step-Audio-EditX Step1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints.layout import MODELS_ROOT
from ...checkpoints.sharded import load_state_dict
from .config import Step1Config


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
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True)
class Step1Checkpoint:
    model_dir: Path
    config: Step1Config
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]

    @property
    def key_count(self) -> int:
        return len(self.state_dict)


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


@dataclass
class LoadedStepAudioEditXModel:
    model_dir: Path
    config: Step1Config
    model: Any  # Step1ForCausalLM
    checkpoint: Step1Checkpoint
    alignment_report: AlignmentReport
    quantization: QuantizationConfig | None = None


class SupportsLoadWeights(Protocol):
    def parameters(self): ...

    def load_weights(self, file_or_weights, strict: bool = True): ...


def _should_skip_key(key: str) -> bool:
    return key.endswith(("rotary_emb.inv_freq", "position_ids"))


def sanitize_state_dict(
    weights: dict[str, mx.array],
    *,
    is_mlx_native: bool = False,
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Normalize checkpoint keys for MLX loading.

    The shipped Step1 checkpoint already matches the target module tree, so
    this mostly preserves keys and skips metadata-only tensors.
    """

    sanitized: dict[str, mx.array] = {}
    skipped: list[str] = []
    renamed: list[tuple[str, str]] = []

    _ = is_mlx_native

    for key, value in weights.items():
        if _should_skip_key(key):
            skipped.append(key)
            continue
        if key in sanitized:
            raise ValueError(f"Duplicate key after sanitization: {key}")
        sanitized[key] = value

    return sanitized, tuple(skipped), tuple(renamed)


def get_quantization_config(config: Step1Config) -> QuantizationConfig | None:
    payload = config.extra.get("quantization")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("`quantization` must be a dict when present in config.json.")
    return QuantizationConfig.from_dict(payload)


def load_step_audio_editx_checkpoint(model_dir: str | Path) -> Step1Checkpoint:
    resolved = Path(model_dir)
    config = Step1Config.from_path(resolved)
    loaded = load_state_dict(resolved)
    is_mlx_native = bool(
        loaded.index is not None and loaded.index.metadata.get("format") == "mlx"
    ) or config.extra.get("quantization") is not None
    state_dict, skipped, renamed = sanitize_state_dict(loaded.weights, is_mlx_native=is_mlx_native)
    return Step1Checkpoint(
        model_dir=resolved,
        config=config,
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped,
        renamed_keys=renamed,
    )


def resolve_step_audio_editx_model_dir(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = False,
) -> Path:
    if model_dir is not None:
        return Path(model_dir)

    base = MODELS_ROOT / "stepfun" / "step_audio_editx"
    int8_dir = base / "mlx-int8"
    original_dir = base / "original"

    if prefer_mlx_int8 and any(int8_dir.glob("*.safetensors")):
        return int8_dir
    if any(original_dir.glob("*.safetensors")):
        return original_dir
    return int8_dir if prefer_mlx_int8 else original_dir


def validate_checkpoint_against_model(
    model: SupportsLoadWeights,
    checkpoint: Step1Checkpoint,
) -> AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    ckpt_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & ckpt_keys):
        m_shape = tuple(int(dim) for dim in model_params[key].shape)
        c_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if m_shape != c_shape:
            shape_mismatches.append((key, m_shape, c_shape))

    return AlignmentReport(
        missing_in_model=tuple(sorted(ckpt_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - ckpt_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_checkpoint_into_model(
    model: SupportsLoadWeights,
    checkpoint: Step1Checkpoint,
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


def quantize_step_audio_editx_model(
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


def save_step_audio_editx_model(
    model: Any,
    model_dir: str | Path,
    *,
    config: Step1Config,
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

    payload = config.to_dict()
    if quantization is not None:
        payload["quantization"] = quantization.to_dict()
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return output_dir


def load_step_audio_editx_model(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = False,
    strict: bool = True,
) -> LoadedStepAudioEditXModel:
    from .model import Step1ForCausalLM

    resolved = resolve_step_audio_editx_model_dir(
        model_dir,
        prefer_mlx_int8=prefer_mlx_int8,
    )
    checkpoint = load_step_audio_editx_checkpoint(resolved)
    model = Step1ForCausalLM(checkpoint.config)
    quantization = get_quantization_config(checkpoint.config)

    if quantization is not None:
        quantize_step_audio_editx_model(
            model,
            quantization,
            state_dict=checkpoint.state_dict,
        )

    alignment = load_checkpoint_into_model(model, checkpoint, strict=strict)
    if hasattr(model, "eval"):
        model.eval()
    mx.eval(model.parameters())

    return LoadedStepAudioEditXModel(
        model_dir=resolved,
        config=checkpoint.config,
        model=model,
        checkpoint=checkpoint,
        alignment_report=alignment,
        quantization=quantization,
    )
