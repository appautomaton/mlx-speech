#!/usr/bin/env python3
"""Convert Step-Audio-EditX Step1 weights to MLX int8 and package runtime assets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_speech.checkpoints import get_stepfun_v4_layouts
from mlx_speech.models.step_audio_editx import (
    QuantizationConfig,
    Step1ForCausalLM,
    load_checkpoint_into_model,
    load_step_audio_editx_checkpoint,
    quantize_step_audio_editx_model,
    save_step_audio_editx_model,
)


def _build_parser() -> argparse.ArgumentParser:
    layouts = get_stepfun_v4_layouts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(layouts.step_audio_editx.original_dir),
        help="Directory containing the original Step-Audio-EditX checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layouts.step_audio_editx.mlx_int8_dir),
        help="Directory to write the converted Step-Audio-EditX checkpoint.",
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument(
        "--skip-supporting-files",
        action="store_true",
        help="Do not copy tokenizer and CosyVoice runtime assets into the output directory.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before writing the converted runtime.",
    )
    return parser


def _copy_runtime_support(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.iterdir()):
        if path.name == "config.json":
            continue
        if path.name == "model.safetensors.index.json":
            continue
        if path.suffix == ".safetensors":
            continue
        destination = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Step-Audio input directory not found: {input_dir}")

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    quantization = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )

    checkpoint = load_step_audio_editx_checkpoint(input_dir)
    model = Step1ForCausalLM(checkpoint.config)
    report = load_checkpoint_into_model(model, checkpoint, strict=True)
    if not report.is_exact_match:
        raise ValueError("Original Step-Audio checkpoint must align exactly before conversion.")

    quantize_step_audio_editx_model(model, quantization, state_dict=checkpoint.state_dict)
    save_step_audio_editx_model(
        model,
        output_dir,
        config=checkpoint.config,
        quantization=quantization,
    )

    if not args.skip_supporting_files:
        _copy_runtime_support(input_dir, output_dir)

    print("Converted Step-Audio-EditX checkpoint")
    print(f"  input_dir: {input_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  quantization: {quantization.mode} {quantization.bits}-bit group={quantization.group_size}")
    print(f"  supporting_files_copied: {not args.skip_supporting_files}")


if __name__ == "__main__":
    main()
