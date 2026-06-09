#!/usr/bin/env python3
"""Package Qwen3-ASR BF16 safetensors for the local MLX runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mlx_speech.models.qwen3_asr.checkpoint import (
    Qwen3ASRConversionReport,
    load_qwen3_asr_checkpoint,
    save_qwen3_asr_bf16_checkpoint,
)


DEFAULT_INPUT = Path("models/qwen3_asr_1_7b/original")
DEFAULT_OUTPUT = Path("models/Qwen3-ASR-1.7B-MLX-BF16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-supporting-files",
        action="store_true",
        help="Only write model.safetensors; do not copy tokenizer/config assets.",
    )
    return parser.parse_args()


def convert_qwen3_asr(
    input_dir: Path,
    output_dir: Path,
    *,
    copy_supporting_files: bool = True,
) -> Qwen3ASRConversionReport:
    checkpoint = load_qwen3_asr_checkpoint(input_dir)
    return save_qwen3_asr_bf16_checkpoint(
        checkpoint,
        output_dir,
        copy_supporting_files=copy_supporting_files,
    )


def main() -> None:
    args = parse_args()
    report = convert_qwen3_asr(
        args.input_dir,
        args.output_dir,
        copy_supporting_files=not args.skip_supporting_files,
    )
    print(f"Input:      {report.input_dir}")
    print(f"Output:     {report.output_dir}")
    print(f"Tensors:    {report.tensor_count}")
    print(f"Renamed:    {len(report.renamed_keys)}")
    print(f"Transposed: {len(report.transposed_keys)}")
    print(f"Skipped:    {len(report.skipped_keys)}")
    print(f"Saved:      {report.output_file}")
    if report.copied_files:
        print(f"Copied:     {len(report.copied_files)} supporting files")


if __name__ == "__main__":
    main()
