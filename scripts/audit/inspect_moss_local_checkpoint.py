#!/usr/bin/env python3
"""Inspect a local MossTTSLocal checkpoint directory."""

from __future__ import annotations

import argparse

from mlx_speech.checkpoints import summarize_prefixes
from mlx_speech.models.moss_local import load_moss_tts_local_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Local checkpoint directory containing config.json and safetensors shards.",
    )
    parser.add_argument(
        "--prefix-depth",
        type=int,
        default=2,
        help="How many dot-separated segments to use for the weight prefix summary.",
    )
    parser.add_argument(
        "--prefix-limit",
        type=int,
        default=25,
        help="Maximum number of prefix rows to print.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_moss_tts_local_checkpoint(args.model_dir)

    print("MossTTSLocal checkpoint summary")
    print(f"  model_dir:        {checkpoint.model_dir}")
    print(f"  source_files:     {len(checkpoint.source_files)}")
    print(f"  tensor_count:     {checkpoint.key_count}")
    print(f"  skipped_keys:     {len(checkpoint.skipped_keys)}")
    print(f"  hidden_size:      {checkpoint.config.hidden_size}")
    print(f"  vocab_size:       {checkpoint.config.vocab_size}")
    print(f"  n_vq:             {checkpoint.config.n_vq}")
    print(f"  sampling_rate:    {checkpoint.config.sampling_rate}")
    print(f"  local_hidden:     {checkpoint.config.local_hidden_size}")
    print(f"  local_layers:     {checkpoint.config.local_num_layers}")

    print("\nTop prefixes:")
    for prefix, count in summarize_prefixes(
        checkpoint.state_dict,
        depth=args.prefix_depth,
    )[: args.prefix_limit]:
        print(f"  {prefix:<50} {count}")


if __name__ == "__main__":
    main()
