from __future__ import annotations

import tomllib
from pathlib import Path

import mlx_speech


def test_package_version_matches_project_metadata() -> None:
    root = Path(__file__).parents[2]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == "0.5.1"
    assert mlx_speech.__version__ == project["project"]["version"]
