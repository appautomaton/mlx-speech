"""Tests for `_resolve_snapshot_dir` quantization-subdir descent."""

from __future__ import annotations

from pathlib import Path

from mlx_speech._hub import _resolve_snapshot_dir


def _make_layout(root: Path, files: list[str]) -> Path:
    """Create empty files at the given relative paths under ``root``."""
    for rel in files:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}", encoding="utf-8")
    return root


def test_returns_root_when_config_present(tmp_path: Path) -> None:
    root = _make_layout(tmp_path, ["config.json"])
    assert _resolve_snapshot_dir(root) == root


def test_descends_into_mlx_int8(tmp_path: Path) -> None:
    root = _make_layout(tmp_path, ["mlx-int8/config.json"])
    assert _resolve_snapshot_dir(root) == root / "mlx-int8"


def test_descends_into_mlx_4bit(tmp_path: Path) -> None:
    root = _make_layout(tmp_path, ["mlx-4bit/config.json"])
    assert _resolve_snapshot_dir(root) == root / "mlx-4bit"


def test_descends_into_mlx_8bit(tmp_path: Path) -> None:
    root = _make_layout(tmp_path, ["mlx-8bit/config.json"])
    assert _resolve_snapshot_dir(root) == root / "mlx-8bit"


def test_priority_int8_over_4bit(tmp_path: Path) -> None:
    root = _make_layout(
        tmp_path,
        ["mlx-int8/config.json", "mlx-4bit/config.json"],
    )
    assert _resolve_snapshot_dir(root) == root / "mlx-int8"


def test_root_config_wins_over_subdir(tmp_path: Path) -> None:
    root = _make_layout(
        tmp_path,
        ["config.json", "mlx-int8/config.json"],
    )
    assert _resolve_snapshot_dir(root) == root


def test_fallback_to_root_when_no_config_anywhere(tmp_path: Path) -> None:
    # Only a weights file exists — no config.json at root or in any known subdir.
    # Resolver returns root so downstream loaders can surface a clearer error.
    root = _make_layout(tmp_path, ["weights.safetensors"])
    assert _resolve_snapshot_dir(root) == root


def test_unknown_subdir_is_not_descended(tmp_path: Path) -> None:
    # Subdir name is not in the known quantization list, so resolver stays put.
    root = _make_layout(tmp_path, ["custom-quant/config.json"])
    assert _resolve_snapshot_dir(root) == root
