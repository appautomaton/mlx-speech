#!/usr/bin/env python3
"""Convert CohereLabs/cohere-transcribe-03-2026 safetensors to MLX int8.

Loads the original checkpoint from --input-dir, sanitizes keys, loads into
the MLX model, quantizes linear layers, and saves the result to --output-dir.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_speech.models.cohere_asr import (
    QuantizationConfig,
    load_cohere_asr_checkpoint,
    load_checkpoint_into_model,
    quantize_cohere_asr_model,
    save_cohere_asr_model,
    CohereAsrForConditionalGeneration,
)


_DEFAULT_INPUT = "models/cohere/cohere_transcribe/original"
_DEFAULT_OUTPUT = "models/cohere/cohere_transcribe/mlx-int8"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default=_DEFAULT_INPUT)
    p.add_argument("--output-dir", default=_DEFAULT_OUTPUT)
    p.add_argument("--bits", type=int, default=8)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--mode", default="affine")
    p.add_argument(
        "--skip-supporting-files",
        action="store_true",
        help="Do not copy tokenizer and config files to the output directory.",
    )
    return p.parse_args()


def _copy_supporting_files(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if not f.is_file():
            continue
        if f.suffix == ".safetensors":
            continue
        if f.name == "config.json":
            continue  # will be written by save_cohere_asr_model
        shutil.copy2(f, dst / f.name)
        print(f"  copied {f.name}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    quantization = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Quant:  {args.bits}-bit {args.mode} group_size={args.group_size}")
    print()

    print("Loading checkpoint...")
    ckpt = load_cohere_asr_checkpoint(input_dir)
    print(f"  {len(ckpt.state_dict)} tensors after sanitization")
    print(f"  {len(ckpt.skipped_keys)} keys skipped")
    print(f"  {len(ckpt.renamed_keys)} keys renamed")

    print("Instantiating model...")
    model = CohereAsrForConditionalGeneration(ckpt.config)

    print("Loading weights (strict)...")
    report = load_checkpoint_into_model(model, ckpt, strict=True)
    print(f"  exact match: {report.is_exact_match}")

    print("Quantizing...")
    model = quantize_cohere_asr_model(model, quantization, state_dict=ckpt.state_dict)

    print(f"Saving to {output_dir} ...")
    save_cohere_asr_model(
        model,
        output_dir,
        config=ckpt.config,
        quantization=quantization,
    )

    if not args.skip_supporting_files:
        print("Copying supporting files...")
        _copy_supporting_files(input_dir, output_dir)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
