#!/usr/bin/env python3
"""Package local Step-Audio tokenizer runtime assets for MLX inference.

This family does not have an int8 conversion path yet. The current Stage 6
"conversion" step packages the local runtime assets into another directory and
verifies that both tokenizer execution paths still load there.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_speech.checkpoints import get_stepfun_v4_layouts
from mlx_speech.models.step_audio_tokenizer import (
    load_step_audio_tokenizer_assets,
    load_step_audio_vq02_model,
    load_step_audio_vq06_model,
)


def _build_parser() -> argparse.ArgumentParser:
    layouts = get_stepfun_v4_layouts()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(layouts.step_audio_tokenizer.original_dir),
        help="Directory containing the local original Step-Audio tokenizer assets.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(layouts.step_audio_tokenizer.mlx_int8_dir),
        help="Directory to write the packaged runtime assets.",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before copying the packaged assets.",
    )
    return parser


def _copy_runtime_assets(source_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.iterdir()):
        destination = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(path, destination)


def main() -> None:
    args = _build_parser().parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Step-Audio tokenizer input directory not found: {input_dir}")

    if args.clean_output and output_dir.exists():
        shutil.rmtree(output_dir)

    print(f"Packaging Step-Audio tokenizer assets from {input_dir} -> {output_dir}")
    _copy_runtime_assets(input_dir, output_dir)

    assets = load_step_audio_tokenizer_assets(output_dir)
    vq02 = load_step_audio_vq02_model(output_dir)
    vq06 = load_step_audio_vq06_model(output_dir)

    print("Packaged Step-Audio tokenizer runtime assets")
    print(f"  output_dir: {output_dir}")
    print(f"  linguistic_codebook_shape: {assets.linguistic_codebook.shape}")
    print(f"  vq02_alignment_exact: {vq02.alignment_report.is_exact_match}")
    print(f"  vq06_alignment_exact: {vq06.alignment_report.is_exact_match}")


if __name__ == "__main__":
    main()
