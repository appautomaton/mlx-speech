#!/usr/bin/env python3
"""Convert VibeVoice Large BF16 safetensors into MLX-native int8 weights.

Quantizes the Qwen2 backbone Linear layers (the bulk of the 9B model).
Keeps conv layers and diffusion head in original precision.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mlx_voice.models.vibevoice.checkpoint import (
    QuantizationConfig,
    load_checkpoint_into_model,
    load_vibevoice_checkpoint,
    quantize_vibevoice_model,
    save_vibevoice_model,
)
from mlx_voice.models.vibevoice.model import VibeVoiceForConditionalGeneration


DEFAULT_INPUT = "models/vibevoice/original"
DEFAULT_OUTPUT = "models/vibevoice/mlx-int8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT,
        help="Directory containing the original VibeVoice-Large checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help="Directory to write the converted MLX checkpoint.",
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument(
        "--skip-supporting-files",
        action="store_true",
        help="Do not copy tokenizer and other non-weight files.",
    )
    return parser.parse_args()


def _copy_supporting_files(source_dir: Path, output_dir: Path) -> list[Path]:
    """Copy non-weight files (tokenizer, preprocessor config, etc.)."""
    copied: list[Path] = []
    for path in sorted(source_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "config.json" or path.name == "model.safetensors.index.json":
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

    print(f"Loading checkpoint from {input_dir}...")
    checkpoint = load_vibevoice_checkpoint(input_dir)

    print("Building model...")
    model = VibeVoiceForConditionalGeneration(checkpoint.config)

    print("Loading weights...")
    report = load_checkpoint_into_model(model, checkpoint, strict=True)
    if not report.is_exact_match:
        raise ValueError("Original checkpoint must align exactly before conversion.")

    print(f"Quantizing ({quantization.mode} {quantization.bits}-bit, group={quantization.group_size})...")
    quantize_vibevoice_model(model, quantization)

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_files: list[Path] = []
    if not args.skip_supporting_files:
        copied_files = _copy_supporting_files(input_dir, output_dir)

    print(f"Saving to {output_dir}...")
    save_vibevoice_model(
        model,
        output_dir,
        config=checkpoint.config,
        quantization=quantization,
    )

    print()
    print("Converted VibeVoice Large checkpoint")
    print(f"  input_dir:  {input_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  quantization: {quantization.mode} {quantization.bits}-bit group={quantization.group_size}")
    print(f"  copied_supporting_files: {len(copied_files)}")


if __name__ == "__main__":
    main()
