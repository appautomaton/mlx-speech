"""Breeze TTS 2 checkpoint layout.

The public inference path decodes with the bundled Qwen audio tokenizer.
Mimi weights stored under ``codec_model.*`` are not part of the MLX package.
Main tensors stay BF16. Codec tensors stay FP32. Convolution kernels are
stored in MLX layout.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

FAMILY = "breeze_tts"
MAIN_DTYPE = "bfloat16"
CODEC_DTYPE = "float32"
SKIPPED_PREFIX = "codec_model."
DEPTH_AUDIO_EMBED = "depth_decoder.model.embed_tokens.weight"
BACKBONE_AUDIO_EMBED = "backbone_model.embed_tokens.embed_audio_tokens.weight"

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


@dataclass(frozen=True)
class BreezeConversion:
    output_dir: Path
    main_tensors: int
    codec_tensors: int
    transposed_codec_tensors: int
    tied_audio_embedding: str


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


def convert_breeze_checkpoint(input_dir: Path, output_dir: Path) -> BreezeConversion:
    """Write the MLX package next to the original snapshot."""

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    codec_input = input_dir / "audio_tokenizer"
    if not (codec_input / "model.safetensors").is_file():
        raise FileNotFoundError(f"Missing bundled audio tokenizer in {codec_input}.")

    main = select_main_weights(_load_weights(input_dir))
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
    _write_config(input_dir / "config.json", output_dir / "config.json")
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


def _write_config(source: Path, destination: Path) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["mlx_speech"] = {
        "family": FAMILY,
        "main_dtype": MAIN_DTYPE,
        "codec_dtype": CODEC_DTYPE,
        "codec": "audio_tokenizer",
    }
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
