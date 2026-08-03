from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_speech._hub import ModelInfo, _DEFAULT_ALLOW_PATTERNS, get_model_path, list_models


def test_dots_aliases_select_gated_int8_and_explicit_base_subdirs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    assert get_model_path("dots-tts-soar") == tmp_path / "soar/mlx-int8"
    assert calls[-1][1]["allow_patterns"] == ["soar/mlx-int8/**", "README.md"]
    assert get_model_path("dots-tts-soar-base") == tmp_path / "soar/mlx-base"
    assert get_model_path("dots-tts-soar-int8") == tmp_path / "soar/mlx-int8"
    assert get_model_path("dots-tts-mf") == tmp_path / "mf/mlx-int8"
    assert calls[-1][1]["allow_patterns"] == ["mf/mlx-int8/**", "README.md"]
    assert get_model_path("dots-tts-mf-base") == tmp_path / "mf/mlx-base"
    assert get_model_path("dots-tts-mf-int8") == tmp_path / "mf/mlx-int8"
    models = list_models("tts")
    assert models["dots-tts-soar"][0] == "appautomaton/dots-tts-mlx"
    assert models["dots-tts-mf-base"][0] == "appautomaton/dots-tts-mlx"
    detailed = list_models("tts", detailed=True)
    assert isinstance(detailed["dots-tts-soar"], ModelInfo)
    assert detailed["dots-tts-soar"].artifact_subdir == "soar/mlx-int8"
    assert detailed["dots-tts-mf-base"].artifact_subdir == "mf/mlx-base"


def test_flat_alias_keeps_default_download_and_resolution(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    assert get_model_path("fish-s2-pro") == tmp_path
    assert calls[0][1]["allow_patterns"] == _DEFAULT_ALLOW_PATTERNS


def test_isolated_cache_materializes_no_sibling_safetensors(monkeypatch, tmp_path: Path) -> None:
    cache = tmp_path / "isolated-cache"
    repository_files = {
        "README.md",
        "soar/mlx-base/core.safetensors",
        "soar/mlx-base/config.json",
        "soar/mlx-base/tokenizer/tokenizer.json",
        "soar/mlx-int8/core.safetensors",
        "mf/mlx-base/core.safetensors",
        "mf/mlx-base/config.json",
        "mf/mlx-base/tokenizer/tokenizer.json",
        "mf/mlx-int8/core.safetensors",
        "mf/mlx-int8/config.json",
        "mf/mlx-int8/tokenizer/tokenizer.json",
    }

    def snapshot_download(repo_id, **kwargs):
        assert repo_id == "appautomaton/dots-tts-mlx"
        selected = kwargs["allow_patterns"][0].removesuffix("/**")
        for relative in repository_files:
            if relative == "README.md" or relative.startswith(f"{selected}/"):
                target = cache / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"selected")
        return str(cache)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    selected = get_model_path("dots-tts-mf")
    assert selected == cache / "mf/mlx-int8"
    materialized = {
        path.relative_to(cache).as_posix() for path in cache.rglob("*") if path.is_file()
    }
    assert materialized == {
        "README.md",
        "mf/mlx-int8/config.json",
        "mf/mlx-int8/core.safetensors",
        "mf/mlx-int8/tokenizer/tokenizer.json",
    }


def test_explicit_shared_repo_selector_downloads_only_requested_artifact(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    selected = get_model_path(
        "appautomaton/dots-tts-mlx",
        artifact_subdir="soar/mlx-base",
    )
    assert selected == tmp_path / "soar/mlx-base"
    assert calls == [
        (
            "appautomaton/dots-tts-mlx",
            {
                "revision": None,
                "allow_patterns": ["soar/mlx-base/**", "README.md"],
                "force_download": False,
            },
        )
    ]


def test_shared_repo_without_selector_fails_before_download(monkeypatch) -> None:
    def snapshot_download(*args, **kwargs):
        raise AssertionError("ambiguous shared repo must fail before download")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    with pytest.raises(ValueError, match="multiple runtime artifacts") as error:
        get_model_path("appautomaton/dots-tts-mlx")
    assert "dots-tts-soar" in str(error.value)
    assert "mf/mlx-int8" in str(error.value)


@pytest.mark.parametrize(
    "artifact_subdir",
    ("", "/soar/mlx-int8", "../mlx-int8", "soar//mlx-int8", "soar\\mlx-int8"),
)
def test_artifact_selector_rejects_unsafe_paths(artifact_subdir: str) -> None:
    with pytest.raises(ValueError, match="artifact_subdir"):
        get_model_path(
            "appautomaton/dots-tts-mlx",
            artifact_subdir=artifact_subdir,
        )


def test_alias_rejects_conflicting_artifact_selector() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        get_model_path("dots-tts-soar", artifact_subdir="mf/mlx-int8")


def test_local_shared_root_uses_explicit_artifact_selector(tmp_path: Path) -> None:
    selected = tmp_path / "mf/mlx-base"
    selected.mkdir(parents=True)
    (selected / "config.json").write_text("{}", encoding="utf-8")
    assert get_model_path(str(tmp_path), artifact_subdir="mf/mlx-base") == selected
