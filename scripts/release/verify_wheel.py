#!/usr/bin/env python3
"""Verify release-wheel metadata and required dots.tts runtime files."""

from __future__ import annotations

import argparse
import email
import zipfile
from pathlib import Path


REQUIRED_DOTS_TTS_FILES = {
    "mlx_speech/generation/dots_tts.py",
    "mlx_speech/models/dots_tts/__init__.py",
    "mlx_speech/tts/_adapters/dots_tts.py",
}


def verify_wheel(path: Path, *, expected_version: str) -> None:
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        metadata_files = sorted(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_files) != 1:
            raise ValueError(
                f"Expected exactly one wheel METADATA file, got {metadata_files}"
            )
        metadata = email.message_from_bytes(archive.read(metadata_files[0]))
    if metadata.get("Name") != "mlx-speech":
        raise ValueError(f"Unexpected wheel name: {metadata.get('Name')!r}")
    if metadata.get("Version") != expected_version:
        raise ValueError(
            f"Wheel version {metadata.get('Version')!r} != {expected_version!r}"
        )
    missing = sorted(REQUIRED_DOTS_TTS_FILES - names)
    if missing:
        raise ValueError(f"Wheel is missing dots.tts runtime files: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    verify_wheel(args.wheel, expected_version=args.version)
    print(f"verified {args.wheel} version={args.version}")


if __name__ == "__main__":
    main()
