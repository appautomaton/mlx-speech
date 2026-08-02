#!/usr/bin/env python3
"""Download and audit immutable, revision-pinned dots.tts source snapshots."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

if not __package__:  # Support the PLAN.md path-based invocation.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.audit.dots_tts_source import (  # noqa: E402
    SourceSpec,
    audit_sources,
    selected_specs,
    write_source_manifest,
)


def verify_remote_info(spec: SourceSpec, info: Any) -> None:
    if info.sha != spec.revision:
        raise ValueError(
            f"{spec.variant} resolved to {info.sha}, expected immutable {spec.revision}"
        )
    if info.id != spec.resolved_repo_id:
        raise ValueError(
            f"{spec.variant} resolved repo {info.id}, expected {spec.resolved_repo_id}"
        )
    siblings = {item.rfilename: item for item in info.siblings}
    if set(siblings) != set(spec.files):
        raise ValueError(
            f"{spec.variant} remote file set changed: "
            f"missing={sorted(set(spec.files) - set(siblings))}, "
            f"unexpected={sorted(set(siblings) - set(spec.files))}"
        )
    for name, pinned in spec.lfs_assets.items():
        sibling = siblings[name]
        lfs = getattr(sibling, "lfs", None) or {}
        sha256 = lfs.get("sha256") if isinstance(lfs, dict) else lfs.sha256
        size = lfs.get("size") if isinstance(lfs, dict) else lfs.size
        if (size, sha256) != (pinned.size, pinned.sha256):
            raise ValueError(
                f"{spec.variant}/{name} remote LFS metadata changed: "
                f"size={size} sha256={sha256}"
            )


def download_snapshot(
    spec: SourceSpec, *, force_download: bool, max_workers: int
) -> None:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as error:  # pragma: no cover - project dependency guard
        raise RuntimeError("dots.tts acquisition requires huggingface_hub") from error

    info = HfApi().model_info(
        spec.repo_id,
        revision=spec.revision,
        files_metadata=True,
    )
    verify_remote_info(spec, info)
    spec.source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=spec.repo_id,
        revision=spec.revision,
        local_dir=spec.source_dir,
        allow_patterns=list(spec.files),
        force_download=force_download,
        max_workers=max_workers,
    )
    # local_dir metadata is downloader state, not part of the immutable source snapshot.
    metadata_dir = spec.source_dir / ".cache" / "huggingface"
    if metadata_dir.exists():
        shutil.rmtree(metadata_dir)
    cache_root = spec.source_dir / ".cache"
    if cache_root.exists() and not any(cache_root.iterdir()):
        cache_root.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("all", "soar", "mf"), default="all")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Audit exact files, hashes, tensor metadata, and latent statistics.",
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    specs = selected_specs(args.variant)
    for spec in specs:
        print(f"downloading {spec.repo_id}@{spec.revision} -> {spec.source_dir}")
        download_snapshot(
            spec,
            force_download=args.force_download,
            max_workers=args.max_workers,
        )
    if args.verify:
        manifest = audit_sources(specs)
        write_source_manifest(manifest)
        for spec in specs:
            print(f"verified immutable {spec.variant} source snapshot")


if __name__ == "__main__":
    main()
