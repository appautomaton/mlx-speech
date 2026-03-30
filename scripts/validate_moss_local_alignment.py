#!/usr/bin/env python3
"""Validate real MossTTSLocal checkpoint keys and shapes against the MLX model."""

from __future__ import annotations

import argparse

from mlx_voice.models.moss_local import (
    MossTTSLocalConfig,
    MossTTSLocalModel,
    load_moss_tts_local_checkpoint,
    validate_checkpoint_against_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default="models/openmoss/moss_tts_local/original",
        help="Local MossTTSLocal checkpoint directory.",
    )
    parser.add_argument(
        "--show-limit",
        type=int,
        default=20,
        help="Maximum number of rows to print from each mismatch section.",
    )
    return parser.parse_args()


def _print_rows(title: str, rows: tuple, limit: int) -> None:
    print(f"\n{title}: {len(rows)}")
    for row in rows[:limit]:
        print(f"  {row}")


def main() -> None:
    args = parse_args()
    checkpoint = load_moss_tts_local_checkpoint(args.model_dir)
    config = MossTTSLocalConfig.from_path(args.model_dir)
    model = MossTTSLocalModel(config)
    report = validate_checkpoint_against_model(model, checkpoint)

    print("MossTTSLocal alignment report")
    print(f"  model_dir: {checkpoint.model_dir}")
    print(f"  exact_match: {report.is_exact_match}")

    _print_rows("checkpoint-only keys", report.missing_in_model, args.show_limit)
    _print_rows("model-only keys", report.missing_in_checkpoint, args.show_limit)
    _print_rows("shape mismatches", report.shape_mismatches, args.show_limit)


if __name__ == "__main__":
    main()
