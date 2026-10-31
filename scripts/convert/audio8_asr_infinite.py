#!/usr/bin/env python3
"""Package Audio8-ASR-Infinite weights for the local MLX runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mlx_speech.models.audio8_asr_infinite.checkpoint import (  # noqa: E402
    Audio8QuantizationConfig,
    load_audio8_checkpoint,
    save_audio8_bf16_checkpoint,
    save_audio8_quantized_checkpoint,
)


DEFAULT_INPUT = Path("models/edge0/audio8_asr_infinite/original")
DEFAULT_BASE_OUTPUT = Path("models/edge0/audio8_asr_infinite")


def convert_audio8_asr_infinite(
    input_dir: Path,
    output_dir: Path,
    *,
    quant: str = "bf16",
    group_size: int = 64,
):
    checkpoint = load_audio8_checkpoint(input_dir)
    if quant == "bf16":
        return save_audio8_bf16_checkpoint(checkpoint, output_dir)
    if quant == "int8":
        return save_audio8_quantized_checkpoint(
            checkpoint,
            output_dir,
            Audio8QuantizationConfig(bits=8, group_size=group_size, mode="affine"),
        )
    raise ValueError(f"Unsupported Audio8 quantization: {quant}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--quant", choices=("bf16", "int8"), default="bf16")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or DEFAULT_BASE_OUTPUT / f"mlx-{args.quant}"
    report = convert_audio8_asr_infinite(
        args.input_dir,
        output_dir,
        quant=args.quant,
        group_size=args.group_size,
    )
    print(f"Saved {report.tensor_count} tensors to {report.output_file}")
    print(f"Quantization: {args.quant}")
    print(f"Conv1D layout conversions: {len(report.transposed_keys)}")


if __name__ == "__main__":
    main()
