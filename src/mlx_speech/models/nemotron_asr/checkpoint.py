"""Strict NeMo checkpoint remapping for Nemotron 3.5 ASR."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ...checkpoints.sharded import load_state_dict
from .config import NemotronASRConfig


class NemotronKeyError(ValueError):
    """Raised when a source checkpoint does not match the known NeMo schema."""


@dataclass(frozen=True)
class QuantizationConfig:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class SourceMapping:
    source: str
    destination: str
    transform: str = "identity"


@dataclass(frozen=True)
class ConversionReport:
    source_count: int
    destination_count: int
    mappings: tuple[SourceMapping, ...]

    @property
    def transformed_count(self) -> int:
        return sum(mapping.transform != "identity" for mapping in self.mappings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_count": self.source_count,
            "destination_count": self.destination_count,
            "transformed_count": self.transformed_count,
            "mappings": [
                {
                    "source": mapping.source,
                    "destination": mapping.destination,
                    "transform": mapping.transform,
                }
                for mapping in self.mappings
            ],
        }


@dataclass(frozen=True)
class AlignmentReport:
    checkpoint_only: tuple[str, ...]
    model_only: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return not self.checkpoint_only and not self.model_only and not self.shape_mismatches


@dataclass(frozen=True)
class NemotronCheckpoint:
    model_dir: Path
    config: NemotronASRConfig
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    conversion_report: ConversionReport | None


_LAYER_KEY = re.compile(r"^encoder\.layers\.(\d+)\.(.+)$")
_LSTM_KEY = re.compile(
    r"^decoder\.prediction\.dec_rnn\.lstm\."
    r"(weight_ih|weight_hh|bias_ih|bias_hh)_l(\d+)$"
)
_CONV_WEIGHT = re.compile(
    r"^(?:encoder\.pre_encode\.conv\.(?:0|2|3|5|6)|"
    r"encoder\.layers\.\d+\.conv\."
    r"(?:pointwise_conv1|depthwise_conv|pointwise_conv2))\.weight$"
)

_LAYER_SUFFIXES = frozenset(
    {
        *(f"{name}.{parameter}" for name in (
            "norm_feed_forward1",
            "norm_self_att",
            "norm_conv",
            "norm_feed_forward2",
            "norm_out",
        ) for parameter in ("weight", "bias")),
        "feed_forward1.linear1.weight",
        "feed_forward1.linear2.weight",
        "feed_forward2.linear1.weight",
        "feed_forward2.linear2.weight",
        "self_attn.pos_bias_u",
        "self_attn.pos_bias_v",
        "self_attn.linear_q.weight",
        "self_attn.linear_k.weight",
        "self_attn.linear_v.weight",
        "self_attn.linear_out.weight",
        "self_attn.linear_pos.weight",
        "conv.pointwise_conv1.weight",
        "conv.depthwise_conv.weight",
        "conv.batch_norm.weight",
        "conv.batch_norm.bias",
        "conv.pointwise_conv2.weight",
    }
)

_FIXED_KEYS = frozenset(
    {
        "preprocessor.featurizer.window",
        "preprocessor.featurizer.fb",
        "encoder.pre_encode.out.weight",
        "encoder.pre_encode.out.bias",
        *(f"encoder.pre_encode.conv.{index}.{parameter}" for index in (0, 2, 3, 5, 6) for parameter in ("weight", "bias")),
        "decoder.prediction.embed.weight",
        "joint.pred.weight",
        "joint.pred.bias",
        "joint.enc.weight",
        "joint.enc.bias",
        "joint.joint_net.2.weight",
        "joint.joint_net.2.bias",
        "prompt_kernel.0.weight",
        "prompt_kernel.0.bias",
        "prompt_kernel.2.weight",
        "prompt_kernel.2.bias",
    }
)


def expected_nemo_keys(n_layers: int = 24, rnn_layers: int = 2) -> frozenset[str]:
    """Return the complete source-key schema for this checkpoint family."""
    keys = set(_FIXED_KEYS)
    for index in range(n_layers):
        keys.update(f"encoder.layers.{index}.{suffix}" for suffix in _LAYER_SUFFIXES)
    for index in range(rnn_layers):
        keys.update(
            f"decoder.prediction.dec_rnn.lstm.{kind}_l{index}"
            for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")
        )
    return frozenset(keys)


def map_nemo_key(
    key: str, *, n_layers: int = 24, rnn_layers: int = 2
) -> SourceMapping:
    """Map one known NeMo key to its MLX destination and transform."""
    if key in _FIXED_KEYS:
        transform = "conv_layout" if _CONV_WEIGHT.fullmatch(key) else "identity"
        return SourceMapping(key, key, transform)

    layer_match = _LAYER_KEY.fullmatch(key)
    if layer_match is not None:
        index, suffix = int(layer_match.group(1)), layer_match.group(2)
        if index < n_layers and suffix in _LAYER_SUFFIXES:
            transform = "conv_layout" if _CONV_WEIGHT.fullmatch(key) else "identity"
            return SourceMapping(key, key, transform)

    lstm_match = _LSTM_KEY.fullmatch(key)
    if lstm_match is not None:
        kind, layer_text = lstm_match.groups()
        layer = int(layer_text)
        if layer < rnn_layers:
            if kind == "weight_ih":
                destination = f"decoder.prediction.dec_rnn.lstm.{layer}.Wx"
                transform = "lstm_input_weight"
            elif kind == "weight_hh":
                destination = f"decoder.prediction.dec_rnn.lstm.{layer}.Wh"
                transform = "lstm_hidden_weight"
            else:
                destination = f"decoder.prediction.dec_rnn.lstm.{layer}.bias"
                transform = "lstm_bias_sum"
            return SourceMapping(key, destination, transform)

    raise NemotronKeyError(f"unmapped NeMo checkpoint key: {key}")


def _as_mlx(value: Any, dtype: mx.Dtype) -> mx.array:
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").float().numpy()
    if isinstance(value, mx.array):
        return value.astype(dtype)
    return mx.array(np.asarray(value)).astype(dtype)


def convert_nemo_state_dict(
    weights: dict[str, Any],
    *,
    dtype: mx.Dtype = mx.bfloat16,
    n_layers: int = 24,
    rnn_layers: int = 2,
) -> tuple[dict[str, mx.array], ConversionReport]:
    """Convert a complete NeMo state dict, rejecting every schema difference."""
    expected = expected_nemo_keys(n_layers, rnn_layers)
    source_keys = set(weights)
    missing = sorted(expected - source_keys)
    extra = sorted(source_keys - expected)
    if missing or extra:
        raise NemotronKeyError(
            "NeMo checkpoint schema mismatch: "
            f"{len(missing)} missing ({missing[:3]}), "
            f"{len(extra)} unmapped ({extra[:3]})"
        )

    converted: dict[str, mx.array] = {}
    mappings: list[SourceMapping] = []
    bias_parts: dict[str, list[mx.array]] = {}
    for key in sorted(weights):
        mapping = map_nemo_key(key, n_layers=n_layers, rnn_layers=rnn_layers)
        tensor_dtype = mx.float32 if key.startswith("preprocessor.") else dtype
        value = _as_mlx(weights[key], tensor_dtype)
        if mapping.transform == "conv_layout":
            if value.ndim == 4:
                value = mx.transpose(value, (0, 2, 3, 1))
            elif value.ndim == 3:
                value = mx.transpose(value, (0, 2, 1))
            else:
                raise NemotronKeyError(
                    f"expected convolution tensor for {key}, got rank {value.ndim}"
                )
        if mapping.transform == "lstm_bias_sum":
            bias_parts.setdefault(mapping.destination, []).append(value)
        elif mapping.destination in converted:
            raise NemotronKeyError(
                f"duplicate destination after remap: {mapping.destination}"
            )
        else:
            converted[mapping.destination] = value
        mappings.append(mapping)

    for destination, parts in bias_parts.items():
        if len(parts) != 2:
            raise NemotronKeyError(
                f"expected two LSTM bias tensors for {destination}, got {len(parts)}"
            )
        if destination in converted:
            raise NemotronKeyError(f"duplicate destination after remap: {destination}")
        converted[destination] = parts[0] + parts[1]

    return converted, ConversionReport(
        source_count=len(weights),
        destination_count=len(converted),
        mappings=tuple(mappings),
    )


def load_nemotron_checkpoint(model_dir: str | Path) -> NemotronCheckpoint:
    """Load an already-converted local MLX checkpoint."""
    resolved = Path(model_dir)
    config = NemotronASRConfig.from_path(resolved)
    loaded = load_state_dict(resolved)
    report_path = resolved / "conversion_report.json"
    report = None
    if report_path.is_file():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        mappings = tuple(SourceMapping(**item) for item in payload["mappings"])
        report = ConversionReport(
            source_count=int(payload["source_count"]),
            destination_count=int(payload["destination_count"]),
            mappings=mappings,
        )
    return NemotronCheckpoint(
        model_dir=resolved,
        config=config,
        state_dict=loaded.weights,
        source_files=loaded.files,
        conversion_report=report,
    )


def validate_state_dict(
    model: nn.Module, state_dict: dict[str, mx.array]
) -> AlignmentReport:
    """Compare a converted state dict to an MLX module tree exactly."""
    parameters = tree_flatten(model.parameters(), destination={})
    model_keys = set(parameters)
    checkpoint_keys = set(state_dict)
    mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(value) for value in parameters[key].shape)
        checkpoint_shape = tuple(int(value) for value in state_dict[key].shape)
        if model_shape != checkpoint_shape:
            mismatches.append((key, model_shape, checkpoint_shape))
    return AlignmentReport(
        checkpoint_only=tuple(sorted(checkpoint_keys - model_keys)),
        model_only=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(mismatches),
    )


def load_state_dict_strict(
    model: nn.Module, state_dict: dict[str, mx.array]
) -> AlignmentReport:
    """Load only after exact key and shape alignment succeeds."""
    report = validate_state_dict(model, state_dict)
    if not report.is_exact_match:
        raise NemotronKeyError(
            "checkpoint alignment failed: "
            f"{len(report.checkpoint_only)} checkpoint-only, "
            f"{len(report.model_only)} model-only, "
            f"{len(report.shape_mismatches)} shape mismatches"
        )
    model.load_weights(list(state_dict.items()), strict=True)
    return report


def quantize_nemotron_model(
    model: nn.Module,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> nn.Module:
    """Quantize eligible Linear/Embedding layers, or recreate a saved layout."""
    checkpoint_keys = set(state_dict) if state_dict is not None else None

    def should_quantize(path: str, module: Any) -> bool:
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


def get_quantization_config(
    config: NemotronASRConfig,
) -> QuantizationConfig | None:
    if config.quantization is None:
        return None
    return QuantizationConfig.from_dict(config.quantization)


__all__ = [
    "AlignmentReport",
    "ConversionReport",
    "NemotronCheckpoint",
    "NemotronKeyError",
    "QuantizationConfig",
    "SourceMapping",
    "convert_nemo_state_dict",
    "expected_nemo_keys",
    "get_quantization_config",
    "load_nemotron_checkpoint",
    "load_state_dict_strict",
    "map_nemo_key",
    "quantize_nemotron_model",
    "validate_state_dict",
]
