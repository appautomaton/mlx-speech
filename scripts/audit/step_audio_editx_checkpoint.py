#!/usr/bin/env python3
"""Inspect the local Step-Audio-EditX Step1 checkpoint and alignment."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_speech.checkpoints.sharded import summarize_prefixes
from mlx_speech.models.step_audio_editx import (
    Step1ForCausalLM,
    load_step_audio_editx_checkpoint,
    validate_checkpoint_against_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a local Step-Audio-EditX Step1 checkpoint.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/stepfun/step_audio_editx/original"),
        help="Checkpoint directory to inspect.",
    )
    parser.add_argument(
        "--prefix-depth",
        type=int,
        default=2,
        help="Depth used for checkpoint prefix summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_step_audio_editx_checkpoint(args.model_dir)
    model = Step1ForCausalLM(checkpoint.config)
    report = validate_checkpoint_against_model(model, checkpoint)

    print("Step-Audio-EditX checkpoint audit")
    print(f"model_dir: {checkpoint.model_dir}")
    print(f"source_files: {len(checkpoint.source_files)}")
    print(f"tensor_count: {checkpoint.key_count}")
    print(f"alignment_exact: {report.is_exact_match}")
    print(
        "config: "
        f"layers={checkpoint.config.num_hidden_layers}, "
        f"hidden={checkpoint.config.hidden_size}, "
        f"heads={checkpoint.config.num_attention_heads}, "
        f"groups={checkpoint.config.num_attention_groups}, "
        f"vocab={checkpoint.config.vocab_size}"
    )
    if checkpoint.skipped_keys:
        print(f"skipped_keys: {len(checkpoint.skipped_keys)}")
    if report.missing_in_model:
        print(f"missing_in_model: {report.missing_in_model[:10]}")
    if report.missing_in_checkpoint:
        print(f"missing_in_checkpoint: {report.missing_in_checkpoint[:10]}")
    if report.shape_mismatches:
        print(f"shape_mismatches: {report.shape_mismatches[:5]}")

    print("top_prefixes:")
    for prefix, count in summarize_prefixes(checkpoint.state_dict, depth=args.prefix_depth)[:20]:
        print(f"  {prefix}: {count}")


if __name__ == "__main__":
    main()
