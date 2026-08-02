#!/usr/bin/env python3
"""Generate the fixed macOS built-in voice clone eval set locally."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="examples/clone_eval/macos_builtin_en.json",
        help="Path to the committed eval manifest.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/clone_eval/macos_builtin_en",
        help="Output directory for generated reference WAV files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing generated reference files.",
    )
    return parser.parse_args()


def load_manifest(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: dict) -> None:
    if manifest.get("schema_version", 1) != 1:
        raise ValueError("unsupported clone eval manifest schema")
    references = manifest.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("clone eval manifest must contain references")
    identifiers = [item.get("id") for item in references]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("clone eval reference ids must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("clone eval reference ids must be unique")
    for item in references:
        for field in ("voice", "reference_text"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"clone eval reference {item['id']} needs {field}")


def audio_record(path: Path) -> dict[str, int | float | str]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frames = handle.getnframes()
    if channels != 1 or sample_rate != 24_000 or sample_width != 2:
        raise ValueError(
            "materialized clone reference must be mono 24 kHz PCM16: "
            f"channels={channels}, sample_rate={sample_rate}, "
            f"sample_width={sample_width}"
        )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "duration_seconds": frames / sample_rate,
    }


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required macOS tool `{name}` was not found in PATH.")


def render_reference(*, voice: str, text: str, output_path: Path) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        aiff_path = temp_dir_path / "reference.aiff"
        subprocess.run(
            ["say", "-v", voice, "-o", str(aiff_path), text],
            check=True,
        )
        subprocess.run(
            [
                "afconvert",
                "-f",
                "WAVE",
                "-d",
                "LEI16@24000",
                "-c",
                "1",
                str(aiff_path),
                str(output_path),
            ],
            check=True,
        )


def main() -> None:
    args = parse_args()
    require_tool("say")
    require_tool("afconvert")

    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    reference_dir = output_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        **manifest,
        "source_manifest": {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
        },
        "materializer": {
            "tool": "macos_say_afconvert",
            "macos_version": platform.mac_ver()[0],
        },
        "references": [],
    }

    for item in manifest["references"]:
        output_path = reference_dir / f"{item['id']}.wav"
        if output_path.exists() and not args.force:
            pass
        else:
            render_reference(
                voice=item["voice"],
                text=item["reference_text"],
                output_path=output_path,
            )

        resolved_item = {**item, **audio_record(output_path)}
        resolved["references"].append(resolved_item)

    resolved_manifest_path = output_dir / "manifest.lock.json"
    with resolved_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print("Materialized clone eval set")
    print(f"  manifest: {args.manifest}")
    print(f"  output_dir: {output_dir}")
    print(f"  references: {len(resolved['references'])}")
    print(f"  lockfile: {resolved_manifest_path}")


if __name__ == "__main__":
    main()
