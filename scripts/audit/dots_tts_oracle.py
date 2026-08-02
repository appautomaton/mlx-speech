#!/usr/bin/env python3
"""Capture and provenance-check bounded dots.tts official-oracle fixtures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[2]
if not __package__:  # Support the PLAN.md path-based invocation.
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from scripts.audit.dots_tts_source import (  # noqa: E402
    SOURCE_MANIFEST,
    SourceSpec,
    selected_specs,
    sha256_file,
)


ROOT = _BOOTSTRAP_ROOT
OFFICIAL_REFERENCE = ROOT / ".references" / "dots.tts"
COMMUNITY_REFERENCE = ROOT / ".references" / "dots-tts-mlx"
WORKER = ROOT / "scripts" / "audit" / "dots_tts_oracle_worker.py"
REQUIREMENTS = ROOT / "scripts" / "audit" / "dots_tts_oracle_requirements.txt"
UV_CACHE = Path("/tmp/mlx-speech-dots-oracle-uv-cache")
SEED = 1729
MAX_ARRAY_BYTES = 16 << 20
MAX_PACK_BYTES = 128 << 20
TOLERANCES = {
    "text_schedule.npz": {"atol": 0.0, "rtol": 0.0},
    "qwen.npz": {"atol": 0.01, "rtol": 0.01},
    "latent_io.npz": {"atol": 1e-5, "rtol": 1e-5},
    "semantic.npz": {"atol": 0.02, "rtol": 0.02},
    "speaker.npz": {"atol": 0.01, "rtol": 0.01},
    "audio_vae.npz": {"atol": 0.02, "rtol": 0.02},
    "dit.npz": {"atol": 0.02, "rtol": 0.02},
    "solver.npz": {"atol": 0.03, "rtol": 0.03},
}
INPUT_CONSTRUCTION = {
    "text": "UTF-8 '[EN]Oracle fixture sentence.' with official tts and interleave templates",
    "qwen": "first six schedule IDs, then one seed+1 standard-normal embedding",
    "latent": "linspace(-2, 2, 4*128), shape [1,4,128]",
    "semantic": "seed+2 standard-normal latent, shape [1,8,128], split 4+4",
    "audio": "48 kHz 0.64 s deterministic 220/440 Hz sine mixture with linear envelope",
    "audio_vae": "first eight hop frames for encode; seed+3 standard-normal [1,128,8] decode latent",
    "dit": "seed+4 standard-normal [1,8,1024], causal mask, positions 0..7, t=duration=0.25",
    "solver": "seed+5 noise [1,4,128]; SOAR 2-step Euler CFG=1.2; MeanFlow NFE=4",
}


def _git_head(path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_manifest() -> dict:
    path = ROOT / SOURCE_MANIFEST
    if not path.is_file():
        raise FileNotFoundError(f"Slice 1 source manifest is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_command(spec: SourceSpec, output: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--isolated",
        "--no-project",
        "--python",
        "3.12",
        "--with-requirements",
        str(REQUIREMENTS),
        "python",
        str(WORKER),
        "--src",
        str((ROOT / spec.source_dir).resolve()),
        "--variant",
        spec.variant,
        "--output",
        str(output.resolve()),
    ]


def _run_worker(spec: SourceSpec, output: Path) -> None:
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHONPATH": str(OFFICIAL_REFERENCE / "src"),
            "TOKENIZERS_PARALLELISM": "false",
            "UV_CACHE_DIR": str(UV_CACHE),
        }
    )
    subprocess.run(oracle_command(spec, output), check=True, cwd=ROOT, env=environment)


def _fixture_inventory(path: Path) -> dict[str, object]:
    arrays: dict[str, object] = {}
    with np.load(path, allow_pickle=False) as payload:
        for name in sorted(payload.files):
            value = payload[name]
            if value.dtype == object:
                raise TypeError(f"{path}:{name} uses forbidden object dtype")
            if value.nbytes > MAX_ARRAY_BYTES:
                raise ValueError(f"{path}:{name} exceeds the bounded fixture limit")
            arrays[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "bytes": int(value.nbytes),
            }
    return {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "tolerance": TOLERANCES[path.name],
        "arrays": arrays,
    }


def _provenance(source: dict) -> dict[str, object]:
    official_commit = _git_head(OFFICIAL_REFERENCE)
    community_commit = _git_head(COMMUNITY_REFERENCE)
    expected_official = "5ed719e3d36f5a3f6d8037ca9a7009d4fd0520ba"
    expected_community = "f64479f51a2a9d7093533732cae86e765d8fb96e"
    if official_commit != expected_official or community_commit != expected_community:
        raise ValueError(
            "reference checkout drift: "
            f"official={official_commit}, community={community_commit}"
        )
    return {
        "references": {
            "official": {"path": ".references/dots.tts", "commit": official_commit},
            "community_mlx": {
                "path": ".references/dots-tts-mlx",
                "commit": community_commit,
            },
        },
        "source_manifest": {
            "path": SOURCE_MANIFEST.as_posix(),
            "sha256": sha256_file(ROOT / SOURCE_MANIFEST),
            "variants": {
                name: {
                    "requested_repo_id": entry["requested_repo_id"],
                    "resolved_repo_id": entry["resolved_repo_id"],
                    "revision": entry["revision"],
                    "file_sha256": {
                        filename: metadata["sha256"]
                        for filename, metadata in entry["files"].items()
                    },
                }
                for name, entry in source["variants"].items()
            },
        },
        "oracle_requirements_sha256": sha256_file(REQUIREMENTS),
        "seed": SEED,
        "input_construction": INPUT_CONSTRUCTION,
    }


def _build_manifest(output: Path, specs: tuple[SourceSpec, ...]) -> dict[str, object]:
    source = _source_manifest()
    provenance = _provenance(source)
    dependencies = None
    fixtures: dict[str, object] = {}
    total_bytes = 0
    for spec in specs:
        variant_dir = output / spec.variant
        worker_metadata_path = variant_dir / "worker_metadata.json"
        worker_metadata = json.loads(worker_metadata_path.read_text(encoding="utf-8"))
        worker_metadata_path.unlink()
        current_dependencies = {
            "python": worker_metadata["python"],
            **worker_metadata["dependencies"],
        }
        if dependencies is None:
            dependencies = current_dependencies
        elif dependencies != current_dependencies:
            raise ValueError("oracle dependencies differ between variant workers")
        actual_names = {path.name for path in variant_dir.iterdir() if path.is_file()}
        if actual_names != set(TOLERANCES):
            raise ValueError(
                f"{spec.variant} fixture set mismatch: expected={sorted(TOLERANCES)}, "
                f"actual={sorted(actual_names)}"
            )
        inventory = {}
        for name in sorted(actual_names):
            item = _fixture_inventory(variant_dir / name)
            total_bytes += item["bytes"]
            inventory[name] = item
        fixtures[spec.variant] = inventory
    if total_bytes > MAX_PACK_BYTES:
        raise ValueError(f"fixture pack is {total_bytes} bytes; limit is {MAX_PACK_BYTES}")
    return {
        "schema_version": 1,
        **provenance,
        "oracle": {
            "environment": "uv --isolated --no-project, Python 3.12, CPU",
            "dependencies": dependencies,
            "capture_command": (
                "uv run python scripts/audit/dots_tts_oracle.py capture "
                "--variant all --output tests/fixtures/dots_tts"
            ),
            "regeneration_command": (
                "uv run python scripts/audit/dots_tts_oracle.py regenerate "
                "--variant all --compare tests/fixtures/dots_tts"
            ),
        },
        "fixtures": fixtures,
    }


def generate_pack(output: Path, specs: tuple[SourceSpec, ...]) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    for spec in specs:
        _run_worker(spec, output / spec.variant)
    manifest = _build_manifest(output, specs)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def validate_fixture_pack(path: Path) -> dict[str, object]:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"fixture manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    current = _provenance(_source_manifest())
    for key, value in current.items():
        if manifest.get(key) != value:
            raise ValueError(f"fixture provenance mismatch for {key}")
    allowed = {"manifest.json"}
    total_bytes = 0
    for variant, files in manifest["fixtures"].items():
        for name, expected in files.items():
            relative = f"{variant}/{name}"
            allowed.add(relative)
            fixture_path = path / relative
            actual = _fixture_inventory(fixture_path)
            if actual != expected:
                raise ValueError(f"fixture inventory/hash mismatch: {relative}")
            total_bytes += actual["bytes"]
    actual_files = {
        item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file()
    }
    if actual_files != allowed:
        raise ValueError(
            f"fixture pack contains unexpected files: {sorted(actual_files - allowed)}"
        )
    if any(Path(name).suffix != ".npz" for name in actual_files - {"manifest.json"}):
        raise ValueError("fixture pack may contain only NPZ arrays and manifest.json")
    if total_bytes > MAX_PACK_BYTES:
        raise ValueError("fixture pack exceeds bounded size")
    return manifest


def compare_fixture_packs(expected: Path, actual: Path) -> None:
    expected_manifest = validate_fixture_pack(expected)
    actual_manifest = validate_fixture_pack(actual)
    if expected_manifest["oracle"] != actual_manifest["oracle"]:
        raise ValueError("oracle dependency or command provenance changed")
    if set(expected_manifest["fixtures"]) != set(actual_manifest["fixtures"]):
        raise ValueError("fixture variants differ")
    for variant, files in expected_manifest["fixtures"].items():
        if set(files) != set(actual_manifest["fixtures"][variant]):
            raise ValueError(f"fixture files differ for {variant}")
        for name, metadata in files.items():
            tolerance = metadata["tolerance"]
            with np.load(expected / variant / name, allow_pickle=False) as left, np.load(
                actual / variant / name, allow_pickle=False
            ) as right:
                if set(left.files) != set(right.files):
                    raise ValueError(f"fixture arrays differ for {variant}/{name}")
                for array_name in left.files:
                    expected_array = left[array_name]
                    actual_array = right[array_name]
                    if expected_array.shape != actual_array.shape:
                        raise ValueError(
                            f"shape mismatch: {variant}/{name}:{array_name}"
                        )
                    if expected_array.dtype != actual_array.dtype:
                        raise ValueError(
                            f"dtype mismatch: {variant}/{name}:{array_name}"
                        )
                    if np.issubdtype(expected_array.dtype, np.inexact):
                        equal = np.allclose(
                            expected_array,
                            actual_array,
                            atol=tolerance["atol"],
                            rtol=tolerance["rtol"],
                            equal_nan=False,
                        )
                    else:
                        equal = np.array_equal(expected_array, actual_array)
                    if not equal:
                        raise ValueError(
                            f"numeric mismatch: {variant}/{name}:{array_name}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--variant", choices=("all", "soar", "mf"), default="all")
    capture.add_argument("--output", type=Path, required=True)
    regenerate = subparsers.add_parser("regenerate")
    regenerate.add_argument(
        "--variant", choices=("all", "soar", "mf"), default="all"
    )
    regenerate.add_argument("--compare", type=Path, required=True)
    args = parser.parse_args()
    specs = selected_specs(args.variant)

    if args.command == "capture":
        output = (ROOT / args.output).resolve()
        if output.exists():
            raise FileExistsError(f"refusing to replace existing fixture pack: {output}")
        generate_pack(output, specs)
        validate_fixture_pack(output)
        print(f"captured official dots.tts fixtures in {output}")
        return

    comparison = (ROOT / args.compare).resolve()
    validate_fixture_pack(comparison)
    with tempfile.TemporaryDirectory(prefix="dots-tts-oracle-") as temp_dir:
        regenerated = Path(temp_dir) / "fixtures"
        generate_pack(regenerated, specs)
        compare_fixture_packs(comparison, regenerated)
    print(f"regenerated fixtures match {comparison} within recorded tolerances")


if __name__ == "__main__":
    main()
