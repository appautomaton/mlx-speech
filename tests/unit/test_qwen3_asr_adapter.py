from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import mlx_speech.asr as asr
import mlx_speech.asr._adapters.qwen3 as adapter_module
from mlx_speech.asr._adapter import ASROutput
from mlx_speech.asr._adapters.qwen3 import Qwen3ASRAdapter
from mlx_speech.asr.generate import add_asr_args, asr_main


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    @classmethod
    def from_dir(cls, model_dir):
        runtime = cls()
        runtime.model_dir = Path(model_dir)
        return runtime

    def transcribe(self, audio, *, sample_rate=16000, language=None, **kwargs):
        self.calls.append((audio, sample_rate, language, kwargs))
        return SimpleNamespace(text="qwen text", language="" if language in (None, "auto") else language)


def test_qwen3_adapter_preserves_auto_language_semantics():
    runtime = _FakeRuntime()
    adapter = Qwen3ASRAdapter(runtime)

    out_default = adapter.generate(np.zeros((4,), dtype=np.float32))
    out_auto = adapter.generate(mx.zeros((4,), dtype=mx.float32), language="auto")
    out_forced = adapter.generate(np.zeros((4,), dtype=np.float32), language="Chinese")

    assert out_default == ASROutput(text="qwen text", language="")
    assert out_auto == ASROutput(text="qwen text", language="")
    assert out_forced == ASROutput(text="qwen text", language="Chinese")
    assert runtime.calls[0][2] is None
    assert runtime.calls[1][2] == "auto"
    assert runtime.calls[2][2] == "Chinese"


def test_asr_load_returns_qwen3_adapter(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_asr"}', encoding="utf-8")
    monkeypatch.setattr(asr, "_get_model_path", lambda path, revision=None: tmp_path)
    monkeypatch.setattr(adapter_module, "Qwen3ASRTranscriber", _FakeRuntime)

    loaded = asr.load(str(tmp_path))

    assert isinstance(loaded, Qwen3ASRAdapter)
    assert loaded._runtime.model_dir == tmp_path


def test_asr_cli_omits_language_by_default_and_passes_auto(monkeypatch, capsys):
    parser = argparse.ArgumentParser()
    add_asr_args(parser)
    default_args = parser.parse_args(["--model", "qwen", "--audio", "sample.wav"])
    auto_args = parser.parse_args(
        ["--model", "qwen", "--audio", "sample.wav", "--language", "auto"]
    )
    calls = []

    class FakeModel:
        def generate(self, audio, *, language=None):
            calls.append((audio, language))
            return ASROutput(text=f"language={language}", language=language or "")

    monkeypatch.setattr(asr, "load", lambda model: FakeModel())

    asr_main(default_args)
    asr_main(auto_args)
    output = capsys.readouterr().out.strip().splitlines()

    assert default_args.language is None
    assert auto_args.language == "auto"
    assert calls == [("sample.wav", None), ("sample.wav", "auto")]
    assert output == ["language=None", "language=auto"]
