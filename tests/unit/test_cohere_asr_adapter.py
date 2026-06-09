from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from mlx_speech.asr._adapter import ASROutput
from mlx_speech.asr._adapters.cohere import CohereASRAdapter


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, *, sample_rate=16000, language="en", **kwargs):
        self.calls.append((np.asarray(audio), sample_rate, language, kwargs))
        return SimpleNamespace(text="cohere text", language=language)


def test_cohere_adapter_maps_omitted_language_to_english():
    runtime = _FakeRuntime()
    adapter = CohereASRAdapter(runtime)

    output = adapter.generate(np.zeros((4,), dtype=np.float32))

    assert output == ASROutput(text="cohere text", language="en")
    assert runtime.calls[0][2] == "en"
