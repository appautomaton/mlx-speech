#!/usr/bin/env python3
"""Convert official FireRedTTS3 Base weights into a flat MLX BF16 artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mlx_speech.models.fireredtts3.checkpoint import convert_fireredtts3  # noqa: E402


DEFAULT_INPUT = Path("models/firered/firered_tts3/original")
DEFAULT_OUTPUT = Path("models/firered/firered_tts3/mlx-bf16")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print("Precision: BF16")
    config = convert_fireredtts3(args.input_dir, args.output_dir)
    print(f"Model type: {config.model_type}")
    for component, filename in config.files.items():
        path = args.output_dir / filename
        print(f"{component}: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
