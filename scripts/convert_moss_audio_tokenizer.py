#!/usr/bin/env python3
"""Convert the upstream Moss audio tokenizer into MLX-native int8 weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_voice.checkpoints import get_openmoss_v0_layouts
from mlx_voice.models.moss_audio_tokenizer import (
    MossAudioTokenizerModel,
    QuantizationConfig,
    load_checkpoint_into_model,
    load_moss_audio_tokenizer_checkpoint,
    quantize_moss_audio_tokenizer_model,
    save_moss_audio_tokenizer_model,
)


def parse_args() -> argparse.Namespace:
    layouts = get_openmoss_v0_layouts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(layouts.audio_tokenizer.original_dir),
        help="Directory containing the original upstream codec checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layouts.audio_tokenizer.mlx_int8_dir),
        help="Directory to write the converted MLX codec checkpoint.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Quantization bit width for eligible MLX layers.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size for eligible MLX layers.",
    )
    parser.add_argument(
        "--mode",
        default="affine",
        help="MLX quantization mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    quantization = QuantizationConfig(
        bits=args.bits,
        group_size=args.group_size,
        mode=args.mode,
    )

    checkpoint = load_moss_audio_tokenizer_checkpoint(input_dir)
    model = MossAudioTokenizerModel(checkpoint.config)
    report = load_checkpoint_into_model(model, checkpoint, strict=True)
    if not report.is_exact_match:
        raise ValueError("Original codec checkpoint must align exactly before conversion.")

    quantize_moss_audio_tokenizer_model(model, quantization)
    save_moss_audio_tokenizer_model(
        model,
        output_dir,
        config=checkpoint.config,
        quantization=quantization,
    )

    print("Converted MossAudioTokenizer checkpoint")
    print(f"  input_dir: {input_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  quantization: {quantization.mode} {quantization.bits}-bit group={quantization.group_size}")


if __name__ == "__main__":
    main()
