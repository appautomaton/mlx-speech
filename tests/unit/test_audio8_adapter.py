from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech.asr._adapters.audio8_asr_infinite import Audio8ASRAdapter


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
