from __future__ import annotations

import json

import pytest

from mlx_speech.asr._registry import _resolve_asr_family


def test_asr_registry_resolves_granite_speech(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "granite_speech"}),
        encoding="utf-8",
    )

    assert _resolve_asr_family(tmp_path) == "granite"


def test_asr_registry_still_resolves_cohere(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "cohere_asr"}),
        encoding="utf-8",
    )

    assert _resolve_asr_family(tmp_path) == "cohere"


def test_asr_registry_resolves_qwen3_asr(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3_asr"}),
        encoding="utf-8",
    )

    assert _resolve_asr_family(tmp_path) == "qwen3"


def test_asr_registry_error_names_supported_families(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "unknown"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cohere_asr, granite_speech, qwen3_asr"):
        _resolve_asr_family(tmp_path)
