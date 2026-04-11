#!/usr/bin/env python3
"""Upload appautomaton MLX model artifacts to Hugging Face.

Usage:
    # Upload one model
    python scripts/hugging_face/upload.py vibevoice

    # Upload multiple models
    python scripts/hugging_face/upload.py vibevoice openmoss-ttsd

    # List available targets
    python scripts/hugging_face/upload.py --list

    # Upload all
    python scripts/hugging_face/upload.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Registry: alias -> (hf_repo_id, local_path, use_large_folder)
#
# use_large_folder=True  → hf upload-large-folder  (chunked, resumable)
# use_large_folder=False → hf upload               (simple single-path)
# ---------------------------------------------------------------------------
MODELS: dict[str, tuple[str, str, bool]] = {
    "cohere-asr": (
        "appautomaton/cohere-asr-mlx",
        "models/cohere/cohere_transcribe/mlx-int8",
        False,
    ),
    "openmoss-audio-tokenizer": (
        "appautomaton/openmoss-audio-tokenizer-mlx",
        "models/openmoss/moss_audio_tokenizer/mlx-int8",
        False,
    ),
    "openmoss-tts-local": (
        "appautomaton/openmoss-tts-local-mlx",
        "models/openmoss/moss_tts_local/mlx-int8",
        False,
    ),
    "openmoss-ttsd": (
        "appautomaton/openmoss-ttsd-mlx",
        "models/openmoss/moss_ttsd/mlx-int8",
        False,
    ),
    "openmoss-sound-effect": (
        "appautomaton/openmoss-sound-effect-mlx",
        "models/openmoss/moss_sound_effect/mlx-4bit",
        True,
    ),
    "vibevoice": (
        "appautomaton/vibevoice-mlx",
        "models/vibevoice/mlx-int8",
        True,
    ),
}


def _resolve_hf() -> str:
    local_hf = Path(sys.executable).with_name("hf")
    if local_hf.exists():
        return str(local_hf)
    return "hf"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        print(f"Error: command exited with {result.returncode}")
        sys.exit(result.returncode)


def upload(alias: str, *, root: Path, hf: str) -> None:
    repo_id, local_rel, large = MODELS[alias]
    local_path = root / local_rel

    if not local_path.exists():
        print(f"Missing: {local_path}")
        sys.exit(1)

    env = os.environ.copy()
    env["HF_HUB_DISABLE_XET"] = "1"

    if large:
        _run(
            [hf, "upload-large-folder", "--repo-type", "model",
             "--num-workers", "1", repo_id, str(local_path)],
            env=env,
        )
    else:
        _run(
            [hf, "upload", "--repo-type", "model",
             repo_id, str(local_path), "mlx-int8"],
            env=env,
        )

    print(f"Done. https://huggingface.co/{repo_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload appautomaton MLX model artifacts to Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "targets",
        nargs="*",
        metavar="MODEL",
        help=f"Model alias(es) to upload. Choices: {', '.join(sorted(MODELS))}",
    )
    parser.add_argument("--all", action="store_true", help="Upload all models.")
    parser.add_argument("--list", action="store_true", help="List available models and exit.")
    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for alias, (repo_id, local, large) in sorted(MODELS.items()):
            mode = "large-folder" if large else "upload"
            print(f"  {alias:<30} → {repo_id}  [{mode}]")
        return

    targets = list(MODELS) if args.all else args.targets
    if not targets:
        parser.print_help()
        sys.exit(1)

    unknown = [t for t in targets if t not in MODELS]
    if unknown:
        print(f"Unknown model(s): {', '.join(unknown)}")
        print(f"Available: {', '.join(sorted(MODELS))}")
        sys.exit(1)

    root = Path(__file__).resolve().parents[2]
    hf = _resolve_hf()

    for alias in targets:
        print(f"--- {alias} ---")
        upload(alias, root=root, hf=hf)


if __name__ == "__main__":
    main()
