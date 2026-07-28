from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_speech.asr as asr
import mlx_speech.asr._adapters.nemotron as adapter_module
from mlx_speech.asr._adapter import ASROutput
from mlx_speech.asr._adapters.nemotron import NemotronASRAdapter


class _FakeSession:
    def feed(self, pcm):  # type: ignore[no-untyped-def]
        return (int(len(pcm)),)

    def finalize(self):  # type: ignore[no-untyped-def]
        return (99,)


class _FakeRuntime:
    def __init__(self) -> None:
        self.calls = []

    @classmethod
    def from_dir(cls, model_dir):  # type: ignore[no-untyped-def]
        runtime = cls()
        runtime.model_dir = Path(model_dir)
        return runtime

    def transcribe(self, audio, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append((audio, kwargs))
        return SimpleNamespace(
            text="nemotron text",
            language=kwargs.get("language") or "auto",
            detected_language="en-US",
        )

    def stream_session(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("stream", kwargs))
        return _FakeSession()


def test_adapter_generate_and_live_session() -> None:
    runtime = _FakeRuntime()
    adapter = NemotronASRAdapter(runtime)  # type: ignore[arg-type]

    output = adapter.generate(
        np.zeros((16,), dtype=np.float32),
        language="auto",
        att_context_size=(56, 3),
    )
    session = adapter.stream_session(language="en-US", att_context_size=[56, 3])

    assert output == ASROutput(text="nemotron text", language="en-US")
    assert session.feed(mx.zeros((137,))) == (137,)
    assert session.finalize() == (99,)
    assert runtime.calls[1] == (
        "stream",
        {"language": "en-US", "att_context_size": [56, 3]},
    )


def test_streaming_adapter_rejects_wrong_sample_rate() -> None:
    adapter = NemotronASRAdapter(_FakeRuntime())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="16000 Hz"):
        adapter.stream_session(sample_rate=48_000)


def test_asr_load_returns_nemotron_adapter(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type": "nemotron_asr"}', encoding="utf-8"
    )
    monkeypatch.setattr(asr, "_get_model_path", lambda path, revision=None: tmp_path)
    monkeypatch.setattr(adapter_module, "NemotronASRModel", _FakeRuntime)

    loaded = asr.load(str(tmp_path))

    assert isinstance(loaded, NemotronASRAdapter)
    assert loaded._runtime.model_dir == tmp_path
