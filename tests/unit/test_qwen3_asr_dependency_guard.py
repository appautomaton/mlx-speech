from __future__ import annotations

from pathlib import Path

import tomllib


def test_qwen3_asr_does_not_add_heavy_runtime_dependencies():
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = {
        dep.split("[", 1)[0].split("<", 1)[0].split(">", 1)[0].split("=", 1)[0].lower()
        for dep in payload["project"]["dependencies"]
    }

    assert "torch" not in dependencies
    assert "torchaudio" not in dependencies
    assert "transformers" not in dependencies
    assert "vllm" not in dependencies
    assert "librosa" not in dependencies
    assert "qwen-asr" not in dependencies


def test_qwen3_asr_package_barrel_stays_lightweight():
    init_path = Path("src/mlx_speech/models/qwen3_asr/__init__.py")
    text = init_path.read_text(encoding="utf-8")

    assert "audio_encoder" not in text
    assert "text_decoder" not in text
    assert "checkpoint" not in text
    assert "model import" not in text
