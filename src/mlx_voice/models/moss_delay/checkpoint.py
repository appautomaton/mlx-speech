"""Checkpoint loading helpers for MossTTSDelay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
from mlx.utils import tree_flatten

from ...checkpoints.layout import MODELS_ROOT
from ...checkpoints.sharded import LoadedStateDict, load_state_dict
from ..moss_local.checkpoint import (
    AlignmentReport,
    QuantizationConfig,
    get_quantization_config,
    load_checkpoint_into_model,
    quantize_moss_tts_local_model,
    sanitize_state_dict,
)
from .config import MossTTSDelayConfig

if TYPE_CHECKING:
    from .model import MossTTSDelayModel


@dataclass(frozen=True)
class MossTTSDelayCheckpoint:
    """Loaded MossTTSDelay checkpoint plus config and loader metadata."""

    model_dir: Path
    config: MossTTSDelayConfig
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]

    @property
    def key_count(self) -> int:
        return len(self.state_dict)


@dataclass
class LoadedMossTTSDelayModel:
    """A fully materialized MLX MossTTSDelay model loaded from a local checkpoint."""

    model_dir: Path
    config: MossTTSDelayConfig
    model: MossTTSDelayModel
    checkpoint: MossTTSDelayCheckpoint
    alignment_report: AlignmentReport
    quantization: QuantizationConfig | None = None


def load_moss_tts_delay_state_dict(model_dir: str | Path) -> LoadedStateDict:
    """Expose the generic loader for scripts that want pre-sanitized delay weights."""

    return load_state_dict(model_dir)


def prepare_moss_tts_delay_runtime_state_dict(
    state_dict: dict[str, mx.array],
    *,
    quantization: QuantizationConfig | None,
) -> dict[str, mx.array]:
    """Prepare runtime tensors for MossTTSDelay.

    Upstream TTSD runs original checkpoints on CPU in float32. Mirror that for the
    unquantized MLX reference path so parity work is not needlessly widened by
    bf16 raw weights. Quantized checkpoints keep their serialized runtime dtypes.
    """

    if quantization is not None:
        return dict(state_dict)

    prepared: dict[str, mx.array] = {}
    for key, value in state_dict.items():
        if mx.issubdtype(value.dtype, mx.floating):
            prepared[key] = value.astype(mx.float32)
        else:
            prepared[key] = value
    return prepared


def load_moss_tts_delay_checkpoint(model_dir: str | Path) -> MossTTSDelayCheckpoint:
    """Load config and sharded safetensors for MossTTSDelay from a local path."""

    resolved_dir = Path(model_dir)
    config = MossTTSDelayConfig.from_path(resolved_dir)
    loaded = load_state_dict(resolved_dir)
    state_dict, skipped_keys, renamed_keys = sanitize_state_dict(loaded.weights)
    return MossTTSDelayCheckpoint(
        model_dir=resolved_dir,
        config=config,
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped_keys,
        renamed_keys=renamed_keys,
    )


def resolve_moss_tts_delay_model_dir(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
) -> Path:
    """Resolve the default local MossTTSDelay path for runtime loading.

    The local default is always the quantized MLX runtime artifact. Use
    ``model_dir`` explicitly if you want to load some other checkpoint path.
    """

    if model_dir is not None:
        return Path(model_dir)

    root_dir = MODELS_ROOT / "openmoss" / "moss_ttsd"
    quantized_dir = root_dir / "mlx-int8"
    _ = prefer_mlx_int8
    if any(quantized_dir.glob("*.safetensors")):
        return quantized_dir
    raise FileNotFoundError(
        "No local quantized MossTTSDelay checkpoint found at "
        f"{quantized_dir}. Pass `model_dir` explicitly to load some other checkpoint."
    )


def quantize_moss_tts_delay_model(
    model: MossTTSDelayModel,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> MossTTSDelayModel:
    """Quantize MossTTSDelay in-place using the MLX module tree."""

    return quantize_moss_tts_local_model(
        model,
        quantization,
        state_dict=state_dict,
    )


def save_moss_tts_delay_model(
    model: MossTTSDelayModel,
    model_dir: str | Path,
    *,
    config: MossTTSDelayConfig,
    quantization: QuantizationConfig | None = None,
) -> Path:
    """Save an MLX-native MossTTSDelay checkpoint directory."""

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


def load_moss_tts_delay_model(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
    strict: bool = True,
) -> LoadedMossTTSDelayModel:
    """Load MossTTSDelay from a local checkpoint directory."""

    from .model import MossTTSDelayModel

    resolved_dir = resolve_moss_tts_delay_model_dir(
        model_dir,
        prefer_mlx_int8=prefer_mlx_int8,
    )
    checkpoint = load_moss_tts_delay_checkpoint(resolved_dir)
    model = MossTTSDelayModel(checkpoint.config)
    quantization = get_quantization_config(checkpoint.config)
    runtime_state_dict = prepare_moss_tts_delay_runtime_state_dict(
        checkpoint.state_dict,
        quantization=quantization,
    )
    if quantization is not None:
        quantize_moss_tts_delay_model(
            model,
            quantization,
            state_dict=runtime_state_dict,
        )
    runtime_checkpoint = MossTTSDelayCheckpoint(
        model_dir=checkpoint.model_dir,
        config=checkpoint.config,
        state_dict=runtime_state_dict,
        source_files=checkpoint.source_files,
        skipped_keys=checkpoint.skipped_keys,
        renamed_keys=checkpoint.renamed_keys,
    )
    alignment_report = load_checkpoint_into_model(model, runtime_checkpoint, strict=strict)
    return LoadedMossTTSDelayModel(
        model_dir=resolved_dir,
        config=checkpoint.config,
        model=model,
        checkpoint=runtime_checkpoint,
        alignment_report=alignment_report,
        quantization=quantization,
    )
