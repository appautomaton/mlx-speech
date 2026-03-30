#!/usr/bin/env python3
"""Download the upstream assets needed by the v0 plan into `models/`."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from mlx_voice.checkpoints.layout import OpenMossV0Layouts, get_openmoss_v0_layouts


TEXT_MODEL_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.tiktoken",
    "*.txt",
]

TEXT_METADATA_PATTERNS = [
    "*.json",
    "*.model",
    "*.tiktoken",
    "*.txt",
]

CODEC_PATTERNS = [
    "*.json",
    "*.safetensors",
]

CODEC_METADATA_PATTERNS = [
    "*.json",
]


def _run_download(
    repo_id: str,
    target_dir: Path,
    patterns: list[str],
    *,
    dry_run: bool,
    force_download: bool,
    max_workers: int,
) -> None:
    command = [
        "hf",
        "download",
        repo_id,
        "--local-dir",
        str(target_dir),
        "--max-workers",
        str(max_workers),
        "--force-download" if force_download else "--no-force-download",
        "--dry-run" if dry_run else "--no-dry-run",
    ]
    for pattern in patterns:
        command.extend(["--include", pattern])
    subprocess.run(command, check=True)


def _download_layout(
    layout,
    *,
    dry_run: bool,
    metadata_only: bool,
    force_download: bool,
    max_workers: int,
) -> None:
    patterns = TEXT_METADATA_PATTERNS if metadata_only else TEXT_MODEL_PATTERNS
    if layout.model_name == "moss_audio_tokenizer":
        patterns = CODEC_METADATA_PATTERNS if metadata_only else CODEC_PATTERNS
    _run_download(
        repo_id=layout.repo_id,
        target_dir=layout.original_dir,
        patterns=patterns,
        dry_run=dry_run,
        force_download=force_download,
        max_workers=max_workers,
    )


def _print_layout_summary(layouts: OpenMossV0Layouts) -> None:
    for asset in (layouts.moss_tts_local, layouts.audio_tokenizer):
        print(f"[{asset.model_name}]")
        print(f"  repo:      {asset.repo_id}")
        print(f"  original:  {asset.original_dir}")
        print(f"  mlx-int8:  {asset.mlx_int8_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        choices=("all", "moss-local", "audio-tokenizer"),
        default="all",
        help="Which upstream asset to download.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download only configs/tokenizers and skip `.safetensors`.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without fetching files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download even if files already exist in the cache.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of download workers to use.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layouts = get_openmoss_v0_layouts().ensure()
    _print_layout_summary(layouts)

    if args.asset in {"all", "moss-local"}:
        _download_layout(
            layouts.moss_tts_local,
            dry_run=args.dry_run,
            metadata_only=args.metadata_only,
            force_download=args.force_download,
            max_workers=args.max_workers,
        )
    if args.asset in {"all", "audio-tokenizer"}:
        _download_layout(
            layouts.audio_tokenizer,
            dry_run=args.dry_run,
            metadata_only=args.metadata_only,
            force_download=args.force_download,
            max_workers=args.max_workers,
        )


if __name__ == "__main__":
    main()
