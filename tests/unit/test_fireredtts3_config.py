from __future__ import annotations

import json

import pytest

from mlx_speech.models.fireredtts3.config import (
    ARTIFACT_FILES,
    FORMAT_VERSION,
    MODEL_TYPE,
    TOKENIZER_FILES,
    FireRedTTS3Config,
)


def _payload() -> dict:
    return {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "precision": "bfloat16",
        "files": dict(ARTIFACT_FILES),
        "tokenizer_files": list(TOKENIZER_FILES),
        "core": {"patch_size": 4, "dtype": "bfloat16"},
        "redae": {
            "audio_sample_rate": 24000,
            "bottleneck_dim": 64,
            "dtype": "bfloat16",
        },
        "qwen": {"hidden_size": 2048},
        "speaker": {"embedding_size": 512},
        "sources": {"weights_revision": "pinned"},
    }


def test_config_loads_flat_artifact_contract(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_payload()), encoding="utf-8")
    config = FireRedTTS3Config.from_dir(tmp_path)
    assert config.model_type == "fireredtts3_base"
    assert config.sample_rate == 24000
    assert config.files == ARTIFACT_FILES


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("model_type", "other", "model_type"),
        ("format_version", 2, "format_version"),
        ("precision", "float16", "precision"),
        ("files", {"core": "model.safetensors"}, "component files"),
    ],
)
def test_config_rejects_incompatible_contract(field, value, match) -> None:
    payload = _payload()
    payload[field] = value
    with pytest.raises(ValueError, match=match):
        FireRedTTS3Config.from_dict(payload)
