from __future__ import annotations

import fnmatch
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_speech._hub import get_model_path, list_models
from mlx_speech.tts._registry import _resolve_tts_family


REPO_ID = "appautomaton/fireredtts3-mlx"
ARTIFACT_SUBDIR = "base/mlx-bf16"


@pytest.mark.parametrize(
    "selector", ("fireredtts3-base", "fireredtts3-base-bf16", REPO_ID)
)
def test_firered_base_download_excludes_instruct_and_original_weights(
    selector: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    repository_files = {
        "README.md": "FireRedTTS3 for MLX",
        "base/mlx-bf16/config.json": '{"model_type": "fireredtts3_base"}',
        "base/mlx-bf16/core.safetensors": "core",
        "base/mlx-bf16/redae.safetensors": "codec",
        "base/mlx-bf16/speaker.safetensors": "speaker",
        "base/mlx-bf16/tokenizer.json": "tokenizer",
        "base/mlx-bf16/tokenizer_config.json": "tokenizer config",
        "base/mlx-bf16/vocab.json": "vocabulary",
        "original/fireredtts3_base/model.safetensors": "FP32 source weights",
        "instruct/mlx-bf16/core.safetensors": "other model",
    }

    def snapshot_download(repo_id, **kwargs):
        assert repo_id == REPO_ID
        patterns = kwargs["allow_patterns"]
        assert patterns == [f"{ARTIFACT_SUBDIR}/**", "README.md"]
        for name, content in repository_files.items():
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
                target = tmp_path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
        return str(tmp_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    kwargs = {"artifact_subdir": ARTIFACT_SUBDIR} if selector == REPO_ID else {}
    selected = get_model_path(selector, **kwargs)
    assert selected == tmp_path / ARTIFACT_SUBDIR
    assert _resolve_tts_family(selected) == "fireredtts3"
    assert not (tmp_path / "instruct").exists()
    assert not (tmp_path / "original").exists()
    assert {path.name for path in selected.iterdir()} == {
        "config.json",
        "core.safetensors",
        "redae.safetensors",
        "speaker.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    if selector != REPO_ID:
        entry = list_models("tts", detailed=True)[selector]
        assert entry.repo_id == REPO_ID
        assert entry.artifact_subdir == ARTIFACT_SUBDIR
