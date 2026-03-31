#!/usr/bin/env python3
"""Convert upstream MOSS-SoundEffect safetensors into MLX-native 4-bit weights."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mlx_voice.checkpoints import INDEX_FILENAME, get_openmoss_v0_layouts
from mlx_voice.models.moss_delay import (
    MossTTSDelayModel,
    QuantizationConfig,
    load_moss_tts_delay_checkpoint,
    quantize_moss_tts_delay_model,
    save_moss_tts_delay_model,
)
from mlx_voice.models.moss_local.checkpoint import load_checkpoint_into_model


def parse_args() -> argparse.Namespace:
    layouts = get_openmoss_v0_layouts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(layouts.moss_sound_effect.original_dir),
        help="Directory containing the upstream MOSS-SoundEffect checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layouts.moss_sound_effect.root_dir / "mlx-4bit"),
        help="Directory to write the converted MLX checkpoint.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bit width for MLX quantized layers.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size for MLX quantized layers.",
    )
    parser.add_argument(
        "--mode",
        default="affine",
        help="MLX quantization mode.",
    )
    parser.add_argument(
        "--skip-supporting-files",
        action="store_true",
        help="Do not copy tokenizer and generation helper files into the output directory.",
    )
    return parser.parse_args()


def _copy_supporting_files(source_dir: Path, output_dir: Path) -> list[Path]:
    copied: list[Path] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "config.json" or path.name == INDEX_FILENAME:
            continue
        if path.suffix == ".safetensors":
            continue
        destination = output_dir / path.name
        shutil.copy2(path, destination)
        copied.append(destination)
    return copied


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    quantization = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )

    checkpoint = load_moss_tts_delay_checkpoint(input_dir)
    model = MossTTSDelayModel(checkpoint.config)
    report = load_checkpoint_into_model(model, checkpoint, strict=True)
    if not report.is_exact_match:
        raise ValueError("Original MOSS-SoundEffect checkpoint must align exactly before conversion.")

    quantize_moss_tts_delay_model(model, quantization)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []
    if not args.skip_supporting_files:
        copied_files = _copy_supporting_files(input_dir, output_dir)
    save_moss_tts_delay_model(
        model,
        output_dir,
        config=checkpoint.config,
        quantization=quantization,
    )

    print("Converted MOSS-SoundEffect checkpoint")
    print(f"  input_dir: {input_dir}")
    print(f"  output_dir: {output_dir}")
    print(
        f"  quantization: {quantization.mode} {quantization.bits}-bit group={quantization.group_size}"
    )
    print(f"  copied_supporting_files: {len(copied_files)}")


if __name__ == "__main__":
    main()
