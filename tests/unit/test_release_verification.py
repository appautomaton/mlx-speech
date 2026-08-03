from __future__ import annotations

import zipfile

import pytest

from scripts.release.verify_version import verify_version
from scripts.release.verify_wheel import REQUIRED_DOTS_TTS_FILES, verify_wheel


def test_verify_version_matches_project_and_runtime(tmp_path) -> None:
    (tmp_path / "src/mlx_speech").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mlx-speech"\nversion = "0.5.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "src/mlx_speech/__init__.py").write_text(
        '__version__ = "0.5.0"\n',
        encoding="utf-8",
    )
    assert verify_version(tmp_path, "v0.5.0") == "0.5.0"


def test_verify_version_rejects_release_ref_mismatch(tmp_path) -> None:
    (tmp_path / "src/mlx_speech").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "mlx-speech"\nversion = "0.5.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "src/mlx_speech/__init__.py").write_text(
        '__version__ = "0.5.0"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Release version mismatch"):
        verify_version(tmp_path, "v0.5.1")


def _write_wheel(path, *, version: str, include_runtime: bool = True) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"mlx_speech-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.4\nName: mlx-speech\nVersion: {version}\n",
        )
        if include_runtime:
            for name in REQUIRED_DOTS_TTS_FILES:
                archive.writestr(name, "")


def test_verify_wheel_requires_version_and_dots_runtime(tmp_path) -> None:
    wheel = tmp_path / "mlx_speech-0.5.0-py3-none-any.whl"
    _write_wheel(wheel, version="0.5.0")
    verify_wheel(wheel, expected_version="0.5.0")


def test_verify_wheel_rejects_missing_dots_runtime(tmp_path) -> None:
    wheel = tmp_path / "mlx_speech-0.5.0-py3-none-any.whl"
    _write_wheel(wheel, version="0.5.0", include_runtime=False)
    with pytest.raises(ValueError, match="missing dots.tts"):
        verify_wheel(wheel, expected_version="0.5.0")
