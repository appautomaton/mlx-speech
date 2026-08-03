#!/usr/bin/env python3
"""Fail when a release ref, project version, and runtime version disagree."""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path


def _runtime_version(init_path: Path) -> str:
    tree = ast.parse(init_path.read_text(encoding="utf-8"), filename=str(init_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"No literal __version__ assignment found in {init_path}")


def verify_version(root: Path, release_ref: str) -> str:
    expected = release_ref.removeprefix("v")
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project_version = str(project["project"]["version"])
    runtime_version = _runtime_version(root / "src/mlx_speech/__init__.py")
    if not expected or expected != project_version or expected != runtime_version:
        raise ValueError(
            "Release version mismatch: "
            f"ref={release_ref!r}, project={project_version!r}, "
            f"runtime={runtime_version!r}"
        )
    return expected


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: verify_version.py <version-or-vtag>")
    root = Path(__file__).resolve().parents[2]
    print(verify_version(root, sys.argv[1]))


if __name__ == "__main__":
    main()
