from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech import asr
from mlx_speech.asr._adapters.audio8_asr_infinite import Audio8ASRAdapter
from scripts.hugging_face.upload import MODELS as HF_UPLOAD_MODELS


class _Runtime:
    def __init__(self) -> None:
        self.calls = []

    def transcribe(self, waveform, *, language, **kwargs):
        self.calls.append((waveform, language, kwargs))
        return SimpleNamespace(text="ok", language=language)


def test_array_input_rejects_mismatched_sample_rate_before_inference():
    runtime = _Runtime()
    adapter = Audio8ASRAdapter(runtime)

    with pytest.raises(ValueError, match="16000 Hz arrays"):
        adapter.generate(np.zeros(4, dtype=np.float32), sample_rate=44100, language="en")

    assert runtime.calls == []


def test_array_input_forwards_language_and_audio_clock():
    runtime = _Runtime()
    adapter = Audio8ASRAdapter(runtime)
    waveform = np.zeros(4, dtype=np.float32)

    result = adapter.generate(waveform, language="zh", frame_len=6, target_delay_ms=480)

    assert result.text == "ok"
    assert result.language == "zh"
    assert runtime.calls[0][1] == "zh"
    assert runtime.calls[0][2] == {"frame_len": 6, "target_delay_ms": 480}


def test_release_aliases_and_upload_targets_are_consistent():
    models = asr.list_models()
    int8 = "appautomaton/audio8-asr-infinite-int8-mlx"
    bf16 = "appautomaton/audio8-asr-infinite-bf16-mlx"

    assert models["audio8-asr-infinite"][0] == int8
    assert models["audio8-asr-infinite-int8"][0] == int8
    assert models["audio8-asr-infinite-bf16"][0] == bf16
    assert HF_UPLOAD_MODELS["audio8-asr-infinite-int8"] == (
        int8, "models/edge0/audio8_asr_infinite/mlx-int8", True
    )
    assert HF_UPLOAD_MODELS["audio8-asr-infinite-bf16"] == (
        bf16, "models/edge0/audio8_asr_infinite/mlx-bf16", True
    )
