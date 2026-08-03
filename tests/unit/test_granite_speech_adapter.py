from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

import mlx_speech.asr as asr
import mlx_speech.asr._adapters.granite_speech as adapter_module
from mlx_speech.asr._adapter import ASROutput
from mlx_speech.asr._adapters.granite_speech import GraniteSpeechASRAdapter
from scripts.hugging_face.upload import MODELS as HF_UPLOAD_MODELS


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    @classmethod
    def from_dir(cls, model_dir):
        runtime = cls()
        runtime.model_dir = Path(model_dir)
        return runtime

    def transcribe(self, audio, *, sample_rate=16000, language="en", **kwargs):
        self.calls.append((np.asarray(audio), sample_rate, language, kwargs))
        return SimpleNamespace(text="granite text", language=language)


def test_granite_adapter_accepts_numpy_and_mx_audio():
    runtime = _FakeRuntime()
    adapter = GraniteSpeechASRAdapter(runtime)

    out_np = adapter.generate(np.zeros((4,), dtype=np.float32))
    out_mx = adapter.generate(mx.zeros((4,), dtype=mx.float32), language="fr")

    assert out_np == ASROutput(text="granite text", language="en")
    assert out_mx == ASROutput(text="granite text", language="fr")
    assert runtime.calls[0][0].dtype == np.float32
    assert runtime.calls[1][0].dtype == np.float32


def test_granite_adapter_accepts_path_audio(monkeypatch, tmp_path):
    runtime = _FakeRuntime()
    adapter = GraniteSpeechASRAdapter(runtime)
    audio_path = tmp_path / "sample.wav"

    def fake_load_audio(path, *, sample_rate, mono):
        assert path == audio_path
        assert sample_rate == 16000
        assert mono is True
        return mx.ones((3,), dtype=mx.float32), 16000

    monkeypatch.setattr(adapter_module, "load_audio", fake_load_audio, raising=False)
    monkeypatch.setattr("mlx_speech.audio.load_audio", fake_load_audio)

    out = adapter.generate(audio_path)

    assert out.text == "granite text"
    assert runtime.calls[0][0].shape == (3,)


def test_asr_load_returns_granite_adapter(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "granite_speech"}', encoding="utf-8")
    monkeypatch.setattr(asr, "_get_model_path", lambda path, revision=None: tmp_path)
    monkeypatch.setattr(adapter_module, "GraniteSpeechAsrModel", _FakeRuntime)

    loaded = asr.load(str(tmp_path))

    assert isinstance(loaded, GraniteSpeechASRAdapter)
    assert loaded._runtime.model_dir == tmp_path


def test_granite_release_aliases_and_upload_target_are_consistent():
    models = asr.list_models()
    repo_id = "appautomaton/granite-4.0-1b-speech-int8-mlx"

    assert models["granite-speech-4.0-1b"][0] == repo_id
    assert models["granite-speech-4.0-1b-int8"][0] == repo_id
    assert HF_UPLOAD_MODELS["granite-speech-4.0-1b-int8"] == (
        repo_id,
        "models/ibm/granite_4_0_1b_speech/mlx-int8",
        True,
    )


def test_granite_model_card_matches_release_contract():
    card = (
        Path(__file__).parents[2]
        / "scripts/hugging_face/model_cards/appautomaton/"
        "granite-4.0-1b-speech-int8-mlx.md"
    ).read_text(encoding="utf-8")

    assert "base_model: ibm-granite/granite-4.0-1b-speech" in card
    assert "base_model_relation: quantized" in card
    assert "group size 64" in card
    assert 'pip install "mlx-speech>=0.5.1"' in card
    assert 'mlx_speech.asr.load("granite-speech-4.0-1b")' in card
    assert "Mandarin speech transcription is not an upstream capability" in card
