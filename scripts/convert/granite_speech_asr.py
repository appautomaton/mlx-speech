#!/usr/bin/env python3
"""Convert IBM Granite Speech 4.0 1B to a selective MLX int8 artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from mlx_speech.models.granite_speech_asr import (
    GraniteSpeechModel,
    QuantizationConfig,
    load_checkpoint_into_model,
    load_granite_speech_checkpoint,
    quantize_granite_speech_model,
    save_granite_speech_model,
)


DEFAULT_INPUT = Path("models/ibm/granite_4_0_1b_speech/original")
DEFAULT_OUTPUT = Path("models/ibm/granite_4_0_1b_speech/mlx-int8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", choices=("affine",), default="affine")
    parser.add_argument(
        "--skip-supporting-files",
        action="store_true",
        help="Write only model.safetensors and config.json.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    quantization = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )

    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(
        f"Quant:  {quantization.mode} {quantization.bits}-bit "
        f"group_size={quantization.group_size}"
    )

    checkpoint = load_granite_speech_checkpoint(args.input_dir)
    model = GraniteSpeechModel(checkpoint.config)
    alignment = load_checkpoint_into_model(model, checkpoint, strict=True)
    if not alignment.is_exact_match:
        raise RuntimeError("Granite Speech source checkpoint did not align exactly")
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())

    quantize_granite_speech_model(model, quantization)
    save_granite_speech_model(
        model,
        args.output_dir,
        config=checkpoint.config,
        quantization=quantization,
        copy_supporting_files_from=(
            None if args.skip_supporting_files else args.input_dir
        ),
    )

    output_file = args.output_dir / "model.safetensors"
    print(f"Saved:   {output_file}")
    print(f"Bytes:   {output_file.stat().st_size}")


if __name__ == "__main__":
    main()
