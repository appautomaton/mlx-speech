"""Breeze TTS 2 checkpoint layout.

The public inference path decodes with the bundled Qwen audio tokenizer.
Mimi weights stored under ``codec_model.*`` are not part of the MLX package.
Main tensors stay BF16. Codec tensors stay FP32. Convolution kernels are
stored in MLX layout. The int8 package quantizes only the per-frame matmuls:
every attention and MLP projection in the backbone and depth decoder layers.
Embeddings, norms, output heads, the depth input projector and the
once-per-utterance text encoder stay BF16.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

FAMILY = "breeze_tts"
MAIN_DTYPE = "bfloat16"
CODEC_DTYPE = "float32"
SKIPPED_PREFIX = "codec_model."
DEPTH_AUDIO_EMBED = "depth_decoder.model.embed_tokens.weight"
BACKBONE_AUDIO_EMBED = "backbone_model.embed_tokens.embed_audio_tokens.weight"

QUANT_PREFIXES = ("backbone_model.layers.", "depth_decoder.model.layers.")
QUANT_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
)

_MAIN_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "LICENSE",
)
_CODEC_FILES = (
    "config.json",
    "configuration.json",
    "preprocessor_config.json",
)


def is_quantizable(path: str) -> bool:
    """Whether int8 targets this module path."""

    return path.startswith(QUANT_PREFIXES) and path.endswith(QUANT_SUFFIXES)


def quantization_paths(weight_names: Iterable[str]) -> tuple[str, ...]:
    """Sorted module paths int8 targets, from weight names."""

    return tuple(
        sorted(
            {
                name.removesuffix(".weight")
                for name in weight_names
                if name.endswith(".weight") and is_quantizable(name.removesuffix(".weight"))
            }
        )
    )


@dataclass(frozen=True)
class BreezeQuantization:
    bits: int
    group_size: int
    mode: str
    paths: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BreezeQuantization":
        config = cls(
            bits=int(payload.get("bits", 0)),
            group_size=int(payload.get("group_size", 0)),
            mode=str(payload.get("mode", "")),
            paths=tuple(str(path) for path in payload.get("paths", ())),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if (self.bits, self.group_size, self.mode) != (8, 64, "affine"):
            raise ValueError("Breeze int8 requires affine 8-bit groups of 64.")
        if not self.paths:
            raise ValueError("Breeze int8 metadata must name every quantized path.")
        if tuple(sorted(set(self.paths))) != self.paths:
            raise ValueError("Breeze quantized paths must be unique and sorted.")
        outside = [path for path in self.paths if not is_quantizable(path)]
        if outside:
            raise ValueError(f"Breeze int8 may not quantize {outside[0]}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "paths": list(self.paths),
        }


def quantize_main_weights(
    weights: dict[str, mx.array], *, bits: int = 8, group_size: int = 64
) -> tuple[dict[str, mx.array], BreezeQuantization]:
    """Pack the targeted projections as affine int8; leave every other tensor."""

    quantization = BreezeQuantization(
        bits=bits, group_size=group_size, mode="affine", paths=quantization_paths(weights)
    )
    quantization.validate()
    packed = dict(weights)
    for path in quantization.paths:
        weight, scales, biases = mx.quantize(
            weights[f"{path}.weight"], group_size=group_size, bits=bits, mode="affine"
        )
        packed[f"{path}.weight"] = weight
        packed[f"{path}.scales"] = scales
        packed[f"{path}.biases"] = biases
    return packed, quantization


def read_quantization(model_dir: Path) -> BreezeQuantization | None:
    """The package's int8 metadata, or None for the BF16 package."""

    payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    marker = payload.get("mlx_speech")
    block = marker.get("quantization") if isinstance(marker, dict) else None
    return None if block is None else BreezeQuantization.from_dict(block)


@dataclass(frozen=True)
class BreezeConversion:
    output_dir: Path
    main_tensors: int
    codec_tensors: int
    transposed_codec_tensors: int
    tied_audio_embedding: str
    quantized_modules: int = 0


def select_main_weights(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    """Drop the unused Mimi codec and materialize the tied audio embedding."""

    selected: dict[str, mx.array] = {}
    for key, value in weights.items():
        if key.startswith(SKIPPED_PREFIX):
            continue
        if value.dtype != mx.bfloat16:
            raise ValueError(f"Main weight {key} is {value.dtype}, expected bfloat16.")
        selected[key] = value
    depth = selected.get(DEPTH_AUDIO_EMBED)
    if depth is None:
        raise ValueError(f"Missing tied audio embedding {DEPTH_AUDIO_EMBED}.")
    existing = selected.get(BACKBONE_AUDIO_EMBED)
    if existing is None:
        selected[BACKBONE_AUDIO_EMBED] = depth
    elif existing.shape != depth.shape:
        raise ValueError(
            "Tied audio embeddings differ: "
            f"{BACKBONE_AUDIO_EMBED} {tuple(existing.shape)} vs "
            f"{DEPTH_AUDIO_EMBED} {tuple(depth.shape)}."
        )
    return selected


def mlx_codec_weight(key: str, value: mx.array) -> tuple[mx.array, bool]:
    """Return one codec tensor in MLX layout, without changing its dtype."""

    if value.dtype != mx.float32:
        raise ValueError(f"Codec weight {key} is {value.dtype}, expected float32.")
    if value.ndim != 3:
        if value.ndim > 3:
            raise ValueError(f"Unexpected codec tensor rank for {key}: {value.ndim}.")
        return value, False
    if _is_conv_transpose_weight(key):
        return value.transpose(1, 2, 0), True
    if key.endswith("conv.weight") or key.endswith("_proj.weight"):
        return value.transpose(0, 2, 1), True
    raise ValueError(f"Unexpected rank-3 codec tensor {key} with shape {tuple(value.shape)}.")


def convert_breeze_checkpoint(
    input_dir: Path, output_dir: Path, *, int8: bool = False
) -> BreezeConversion:
    """Write the MLX package next to the original snapshot; ``int8`` packs the
    per-frame projections."""

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    codec_input = input_dir / "audio_tokenizer"
    if not (codec_input / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing bundled audio tokenizer in {codec_input}.")

    main = select_main_weights(_load_weights(input_dir))
    quantization = None
    if int8:
        main, quantization = quantize_main_weights(main)
    codec: dict[str, mx.array] = {}
    transposed: list[str] = []
    for key, value in _load_weights(codec_input).items():
        converted, did_transpose = mlx_codec_weight(key, value)
        codec[key] = converted
        if did_transpose:
            transposed.append(key)
    mx.eval(*main.values(), *codec.values())

    output_dir.mkdir(parents=True)
    codec_output = output_dir / "audio_tokenizer"
    codec_output.mkdir()
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        main,
        metadata={"format": "mlx", "dtype": MAIN_DTYPE},
    )
    mx.save_safetensors(
        str(codec_output / "model.safetensors"),
        codec,
        metadata={"format": "mlx", "dtype": CODEC_DTYPE},
    )
    _write_config(input_dir / "config.json", output_dir / "config.json", quantization)
    for name in _MAIN_FILES:
        source = input_dir / name
        if source.is_file():
            shutil.copy2(source, output_dir / name)
    for name in _CODEC_FILES:
        source = codec_input / name
        if source.is_file():
            shutil.copy2(source, codec_output / name)
    return BreezeConversion(
        output_dir=output_dir,
        main_tensors=len(main),
        codec_tensors=len(codec),
        transposed_codec_tensors=len(transposed),
        tied_audio_embedding=BACKBONE_AUDIO_EMBED,
        quantized_modules=0 if quantization is None else len(quantization.paths),
    )


def _is_conv_transpose_weight(key: str) -> bool:
    return (
        key.startswith("decoder.decoder.") and key.endswith(".block.1.conv.weight")
    ) or (".upsample." in key and key.endswith(".0.conv.weight"))


def _load_weights(model_dir: Path) -> dict[str, mx.array]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {model_dir}.")
    weights: dict[str, mx.array] = {}
    for path in files:
        weights.update(mx.load(str(path)))
    return weights


def _write_config(
    source: Path, destination: Path, quantization: BreezeQuantization | None = None
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["mlx_speech"] = {
        "family": FAMILY,
        "main_dtype": MAIN_DTYPE,
        "codec_dtype": CODEC_DTYPE,
        "codec": "audio_tokenizer",
    }
    if quantization is not None:
        payload["mlx_speech"]["quantization"] = quantization.to_dict()
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
