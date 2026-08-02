#!/usr/bin/env python3
"""Audit pinned dots.tts source snapshots without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open


SOURCE_ROOT = Path("models/dots_tts")
SOURCE_MANIFEST = SOURCE_ROOT / "source_manifest.json"

SOURCE_FILES = (
    ".gitattributes",
    "README.md",
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "latent_stats.pt",
    "llm_config.json",
    "merges.txt",
    "model.safetensors",
    "speaker_encoder.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "vocoder.safetensors",
)


@dataclass(frozen=True)
class LfsAsset:
    size: int
    sha256: str


@dataclass(frozen=True)
class SourceSpec:
    variant: str
    repo_id: str
    resolved_repo_id: str
    revision: str
    source_dir: Path
    files: tuple[str, ...] = SOURCE_FILES
    lfs_assets: dict[str, LfsAsset] = field(default_factory=dict)


_COMMON_LFS = {
    "latent_stats.pt": LfsAsset(
        3_197, "313b13af56d659ecf869d5f854508fcf823c8f957aefc6bc05244991abd6ffe1"
    ),
    "speaker_encoder.safetensors": LfsAsset(
        29_150_484,
        "1cf3861c9dee79e4db34bd0b8a4155e68bed27a7c6274e168bb6ee4fed191c85",
    ),
    "tokenizer.json": LfsAsset(
        11_423_263,
        "c16521f66774c7a4774e5303b7c8ec5c99830c0be5aef6c6edde3ca2a5e05dd0",
    ),
    "vocoder.safetensors": LfsAsset(
        723_585_584,
        "c0e45c08f480df67ac4c354b465355fcc7e2f6c8765263b6dfeddd1f4671c93d",
    ),
}

SOURCE_SPECS = {
    "soar": SourceSpec(
        variant="soar",
        repo_id="rednote-hilab/dots.tts-soar",
        resolved_repo_id="dots-studio/dots.tts-soar",
        revision="e3520f75254d0020a0406db31c51a79d00d22d55",
        source_dir=SOURCE_ROOT / "soar" / "original",
        lfs_assets={
            **_COMMON_LFS,
            "model.safetensors": LfsAsset(
                4_396_289_197,
                "2787e6d4fe0b27ac33d28072abcadef53802c841bff037f3ee1f6bb5e1d3a2ce",
            ),
        },
    ),
    "mf": SourceSpec(
        variant="mf",
        repo_id="rednote-hilab/dots.tts-mf",
        resolved_repo_id="dots-studio/dots.tts-mf",
        revision="25c53fb462e57087e52237daa5ea30df1c5cc328",
        source_dir=SOURCE_ROOT / "mf" / "original",
        lfs_assets={
            **_COMMON_LFS,
            "model.safetensors": LfsAsset(
                4_398_915_254,
                "a16d5798da197bf647fc01915236873e4672e975b0341360703ec49d002c4696",
            ),
        },
    ),
}


def selected_specs(variant: str) -> tuple[SourceSpec, ...]:
    if variant == "all":
        return tuple(SOURCE_SPECS[name] for name in ("soar", "mf"))
    try:
        return (SOURCE_SPECS[variant],)
    except KeyError as error:
        raise ValueError(f"unsupported dots.tts source variant: {variant}") from error


def sha256_file(path: Path, *, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    """Accept the two NumPy arrays in latent_stats.pt and nothing executable."""

    _ALLOWED_GLOBALS = {
        ("_codecs", "encode"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"forbidden latent-stats global: {module}.{name}")
        return super().find_class(module, name)

    def persistent_load(self, pid: object) -> Any:
        raise pickle.UnpicklingError(f"persistent IDs are forbidden: {pid!r}")


def _latent_pickle_bytes(path: Path) -> bytes:
    if not zipfile.is_zipfile(path):
        return path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        records = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(records) != 1:
            raise ValueError(f"expected one data.pkl in {path}, found {records}")
        return archive.read(records[0])


def read_latent_stats(path: Path) -> dict[str, np.ndarray]:
    payload = _RestrictedNumpyUnpickler(io.BytesIO(_latent_pickle_bytes(path))).load()
    if not isinstance(payload, dict) or set(payload) != {"mean", "var"}:
        raise ValueError("latent_stats.pt must contain exactly mean and var")
    arrays: dict[str, np.ndarray] = {}
    for name in ("mean", "var"):
        value = payload[name]
        if not isinstance(value, np.ndarray):
            raise TypeError(f"latent statistic {name} must be a NumPy array")
        if value.dtype != np.float32:
            raise TypeError(f"latent statistic {name} must be float32, got {value.dtype}")
        if not value.flags.c_contiguous:
            raise ValueError(f"latent statistic {name} must be C-contiguous")
        arrays[name] = value
    if arrays["mean"].shape != arrays["var"].shape:
        raise ValueError("latent mean and variance shapes differ")
    return arrays


def safetensors_inventory(path: Path) -> dict[str, object]:
    dtypes: Counter[str] = Counter()
    tensor_count = 0
    element_count = 0
    with safe_open(path, framework="numpy") as handle:
        for key in handle.keys():
            tensor = handle.get_slice(key)
            shape = tuple(tensor.get_shape())
            dtypes[str(tensor.get_dtype())] += 1
            tensor_count += 1
            element_count += int(np.prod(shape, dtype=np.int64))
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "dtypes": dict(sorted(dtypes.items())),
    }


def _source_files(source_dir: Path) -> set[str]:
    return {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    }


def audit_variant(spec: SourceSpec) -> dict[str, object]:
    actual = _source_files(spec.source_dir) if spec.source_dir.is_dir() else set()
    expected = set(spec.files)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{spec.variant} source layout mismatch: missing={missing}, unexpected={unexpected}"
        )

    files: dict[str, object] = {}
    for name in spec.files:
        path = spec.source_dir / name
        size = path.stat().st_size
        digest = sha256_file(path)
        if pinned := spec.lfs_assets.get(name):
            if (size, digest) != (pinned.size, pinned.sha256):
                raise ValueError(
                    f"{spec.variant}/{name} does not match pinned LFS asset: "
                    f"got size={size} sha256={digest}"
                )
        entry: dict[str, object] = {"bytes": size, "sha256": digest}
        if path.suffix == ".safetensors":
            entry["safetensors"] = safetensors_inventory(path)
        files[name] = entry

    latent = read_latent_stats(spec.source_dir / "latent_stats.pt")
    latent_inventory = {
        name: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        for name, value in latent.items()
    }
    return {
        "requested_repo_id": spec.repo_id,
        "resolved_repo_id": spec.resolved_repo_id,
        "revision": spec.revision,
        "source_dir": spec.source_dir.as_posix(),
        "files": files,
        "latent_stats": latent_inventory,
    }


def audit_sources(specs: tuple[SourceSpec, ...]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "variants": {spec.variant: audit_variant(spec) for spec in specs},
    }


def write_source_manifest(payload: dict[str, object], path: Path = SOURCE_MANIFEST) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", *SOURCE_SPECS), default="all")
    parser.add_argument("--output", type=Path, default=SOURCE_MANIFEST)
    args = parser.parse_args()
    specs = selected_specs(args.variant)
    manifest = audit_sources(specs)
    write_source_manifest(manifest, args.output)
    for spec in specs:
        file_count = len(manifest["variants"][spec.variant]["files"])
        print(f"verified dots.tts {spec.variant}: {file_count} files at {spec.revision}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
