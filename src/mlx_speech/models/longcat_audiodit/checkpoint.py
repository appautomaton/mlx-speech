"""Checkpoint loading helpers for LongCat AudioDiT."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints.layout import MODELS_ROOT
from ...checkpoints.sharded import load_state_dict
from .config import LongCatAudioDiTConfig
from .model import LongCatAudioDiTModel
from .text_encoder import LongCatUMT5Encoder
from .transformer import LongCatAudioDiTTransformer
from .vae import LongCatAudioDiTVae


@dataclass(frozen=True)
class LongCatCheckpoint:
    model_dir: Path
    config: LongCatAudioDiTConfig
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
class LoadedLongCatModel:
    model_dir: Path
    config: LongCatAudioDiTConfig
    model: Any
    checkpoint: LongCatCheckpoint
    alignment_report: AlignmentReport
    quantization: Any = None


class SupportsLoadWeights(Protocol):
    def parameters(self): ...

    def load_weights(self, file_or_weights, strict: bool = True): ...


def resolve_longcat_model_dir(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
) -> Path:
    if model_dir is not None:
        return Path(model_dir)

    base = MODELS_ROOT / "longcat_audiodit"
    int8_dir = base / "mlx-int8"
    original_dir = base / "original"

    if prefer_mlx_int8 and any(int8_dir.glob("*.safetensors")):
        return int8_dir
    if any(original_dir.glob("*.safetensors")):
        return original_dir
    return int8_dir if prefer_mlx_int8 else original_dir


def resolve_longcat_tokenizer_dir(tokenizer_dir: str | Path | None = None) -> Path:
    if tokenizer_dir is not None:
        return Path(tokenizer_dir)
    return MODELS_ROOT / "longcat_audiodit" / "tokenizer" / "umt5-base"


def _should_skip_key(key: str) -> bool:
    return key.endswith(("rotary_emb.inv_freq", "position_ids"))


def _fold_weight_norm(weight_g: mx.array, weight_v: mx.array) -> mx.array:
    norm = mx.sqrt(
        mx.sum(mx.square(weight_v.astype(mx.float32)), axis=(1, 2), keepdims=True)
    )
    return (weight_g.astype(mx.float32) * weight_v.astype(mx.float32)) / norm


def _to_mlx_conv1d(weight: mx.array) -> mx.array:
    return mx.transpose(weight, (0, 2, 1))


def _to_mlx_convtranspose1d(weight: mx.array) -> mx.array:
    return mx.transpose(weight, (1, 2, 0))


_DECODER_TRANSPOSE_RE = re.compile(r"^vae\.decoder\.layers\.\d+\.layers\.1\.weight$")


def _is_decoder_transpose_key(key: str) -> bool:
    return bool(_DECODER_TRANSPOSE_RE.fullmatch(key))


def sanitize_state_dict(
    weights: dict[str, mx.array],
    *,
    is_mlx_native: bool,
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...]]:
    if is_mlx_native:
        return weights, (), ()

    sanitized: dict[str, mx.array] = {}
    skipped: list[str] = []
    renamed: list[tuple[str, str]] = []
    consumed: set[str] = set()

    for key, value in weights.items():
        if key in consumed:
            continue
        if _should_skip_key(key):
            skipped.append(key)
            continue

        if key.endswith("weight_g"):
            partner = f"{key[:-8]}weight_v"
            if partner not in weights:
                raise KeyError(f"Missing weight_norm partner for {key}: {partner}")
            folded = _fold_weight_norm(value, weights[partner])
            target = f"{key[:-8]}weight"
            if _is_decoder_transpose_key(target):
                folded = _to_mlx_convtranspose1d(folded)
            else:
                folded = _to_mlx_conv1d(folded)
            sanitized[target] = folded
            renamed.append((key, target))
            consumed.add(key)
            consumed.add(partner)
            continue

        if key.endswith("weight_v"):
            if f"{key[:-8]}weight_g" in weights:
                continue

        if value.ndim == 3 and key.endswith("weight"):
            if _is_decoder_transpose_key(key):
                value = _to_mlx_convtranspose1d(value)
            else:
                value = _to_mlx_conv1d(value)

        sanitized[key] = value

    return sanitized, tuple(skipped), tuple(renamed)


def load_longcat_checkpoint(model_dir: str | Path) -> LongCatCheckpoint:
    resolved = Path(model_dir)
    config = LongCatAudioDiTConfig.from_path(resolved)
    loaded = load_state_dict(resolved)
    is_mlx_native = (
        bool(loaded.index is not None and loaded.index.metadata.get("format") == "mlx")
        or config.quantization is not None
    )
    state_dict, skipped, renamed = sanitize_state_dict(
        loaded.weights,
        is_mlx_native=is_mlx_native,
    )
    return LongCatCheckpoint(
        model_dir=resolved,
        config=config,
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped,
        renamed_keys=renamed,
    )


def validate_checkpoint_against_model(
    model: SupportsLoadWeights,
    checkpoint: LongCatCheckpoint,
) -> AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    ckpt_keys = set(checkpoint.state_dict)
    mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & ckpt_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        ckpt_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != ckpt_shape:
            mismatches.append((key, model_shape, ckpt_shape))
    return AlignmentReport(
        missing_in_model=tuple(sorted(ckpt_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - ckpt_keys)),
        shape_mismatches=tuple(mismatches),
    )


def load_checkpoint_into_model(
    model: SupportsLoadWeights,
    checkpoint: LongCatCheckpoint,
    *,
    strict: bool = True,
) -> AlignmentReport:
    report = validate_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(f"Checkpoint/model alignment failed: {report}")
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return report


def quantize_longcat_model(
    model: Any,
    quantization,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> Any:
    quantized_keys = set(state_dict) if state_dict is not None else None

    blocked_prefixes = (
        "transformer.input_embed",
        "transformer.text_embed",
        "transformer.latent_embed",
        "transformer.latent_cond_embedder",
    )

    def should_quantize(path: str, module: Any) -> bool:
        if not (hasattr(module, "weight") and hasattr(module, "to_quantized")):
            return False
        if module.weight.shape[-1] % quantization.group_size != 0:
            return False
        if path.startswith(blocked_prefixes):
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


def save_longcat_model(
    model: SupportsLoadWeights,
    model_dir: str | Path,
    *,
    config: LongCatAudioDiTConfig,
    quantization=None,
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
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return output_dir


def load_longcat_model(
    model_dir: str | Path | None = None,
    *,
    prefer_mlx_int8: bool = True,
    strict: bool = True,
) -> LoadedLongCatModel:
    resolved = resolve_longcat_model_dir(model_dir, prefer_mlx_int8=prefer_mlx_int8)
    checkpoint = load_longcat_checkpoint(resolved)
    model = LongCatAudioDiTModel(
        checkpoint.config,
        text_encoder=LongCatUMT5Encoder(checkpoint.config.text_encoder_config),
        transformer=LongCatAudioDiTTransformer(checkpoint.config),
        vae=LongCatAudioDiTVae(checkpoint.config.vae_config),
    )
    quantization = checkpoint.config.quantization
    if quantization is not None:
        quantize_longcat_model(model, quantization, state_dict=checkpoint.state_dict)
    alignment = load_checkpoint_into_model(model, checkpoint, strict=strict)
    return LoadedLongCatModel(
        model_dir=resolved,
        config=checkpoint.config,
        model=model,
        checkpoint=checkpoint,
        alignment_report=alignment,
        quantization=quantization,
    )
