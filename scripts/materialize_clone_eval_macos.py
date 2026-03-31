#!/usr/bin/env python3
"""Generate the fixed macOS built-in voice clone eval set locally."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
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
    output_dir = Path(args.output_dir)
    reference_dir = output_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "name": manifest["name"],
        "language": manifest["language"],
        "description": manifest.get("description"),
        "references": [],
        "prompts": manifest["prompts"],
        "recommended_presets": manifest.get("recommended_presets", []),
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

        resolved_item = dict(item)
        resolved_item["path"] = str(output_path)
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
