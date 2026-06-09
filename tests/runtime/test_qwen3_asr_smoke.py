from __future__ import annotations

from pathlib import Path

import pytest

import mlx_speech.asr as asr


MODEL_DIR = Path("models/Qwen3-ASR-1.7B-MLX-BF16")
SAMPLES = {
    "English": Path("models/Qwen3-ASR-1.7B-MLX-BF16/samples/english.wav"),
    "Chinese": Path("models/Qwen3-ASR-1.7B-MLX-BF16/samples/chinese.wav"),
    "Mixed": Path("models/Qwen3-ASR-1.7B-MLX-BF16/samples/mixed.wav"),
}


pytestmark = pytest.mark.runtime


def _require_local_assets() -> None:
    missing = []
    if not (MODEL_DIR / "config.json").exists() or not (MODEL_DIR / "model.safetensors").exists():
        missing.append(str(MODEL_DIR))
    missing_samples = [str(path) for path in SAMPLES.values() if not path.exists()]
    missing.extend(missing_samples)
    if missing:
        pytest.skip("Qwen3-ASR local runtime smoke assets not present: " + ", ".join(missing))


def test_qwen3_asr_local_runtime_transcribes_english_chinese_and_mixed_audio():
    _require_local_assets()
    model = asr.load(str(MODEL_DIR))

    results = {
        name: model.generate(path, max_new_tokens=256)
        for name, path in SAMPLES.items()
    }

    for result in results.values():
        assert result.text.strip()
        assert len(result.tokens) < 256

    assert results["English"].language != "en"
