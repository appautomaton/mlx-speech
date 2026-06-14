"""DramaBox first-class TTS alias: registration, family resolution, adapter,
and stereo WAV output. None of these require model weights."""

import json
import wave

import mlx.core as mx
import pytest

from mlx_speech._hub import DRAMABOX_GEMMA_REPO
from mlx_speech.audio import write_wav
from mlx_speech.tts import list_models
from mlx_speech.tts._adapters.dramabox import DramaBoxAdapter
from mlx_speech.tts._registry import _resolve_tts_family


def test_dramabox_alias_registered():
    models = list_models()
    assert "dramabox" in models
    repo, _desc = models["dramabox"]
    assert repo == "appautomaton/dramabox-tts-3.3b-bf16-mlx"


def test_dramabox_gemma_backbone_default_repo():
    assert DRAMABOX_GEMMA_REPO == "appautomaton/gemma-3-12b-it-backbone-4bit-mlx"


def test_resolve_dramabox_family_from_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "dramabox-tts"}))
    assert _resolve_tts_family(tmp_path) == "dramabox"


class _FakeResult:
    def __init__(self):
        self.waveform = mx.zeros((2, 8), dtype=mx.float32)
        self.sample_rate = 48000


class _FakeModel:
    def __init__(self):
        self.calls = []

    def generate(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return _FakeResult()


def test_adapter_maps_duration_and_passthrough_kwargs():
    fake = _FakeModel()
    out = DramaBoxAdapter(fake).generate(
        "hello", duration_seconds=3.0, cfg_scale=4.0, steps=20, seed=7
    )
    text, kwargs = fake.calls[0]
    assert text == "hello"
    assert kwargs == {"duration_s": 3.0, "cfg_scale": 4.0, "steps": 20, "seed": 7}
    assert out.sample_rate == 48000
    assert out.waveform.shape == (2, 8)


def test_adapter_maps_reference_audio_to_voice_ref():
    fake = _FakeModel()
    DramaBoxAdapter(fake).generate("hi", reference_audio="ref.wav")
    _text, kwargs = fake.calls[0]
    assert kwargs["voice_ref"] == "ref.wav"


def test_adapter_requires_text():
    with pytest.raises(ValueError, match="non-empty text"):
        DramaBoxAdapter(_FakeModel()).generate(None)


def test_adapter_rejects_mx_array_reference():
    with pytest.raises(TypeError, match="file path, not mx.array"):
        DramaBoxAdapter(_FakeModel()).generate("hi", reference_audio=mx.zeros((4,)))


def test_write_wav_stereo_roundtrip(tmp_path):
    stereo = mx.stack(
        [mx.zeros((16,), dtype=mx.float32), mx.full((16,), 0.5, dtype=mx.float32)],
        axis=0,
    )  # [2, 16]
    path = write_wav(tmp_path / "stereo.wav", stereo, sample_rate=48000)
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 2
        assert f.getframerate() == 48000
        assert f.getnframes() == 16


def test_write_wav_mono_unchanged(tmp_path):
    path = write_wav(tmp_path / "mono.wav", mx.zeros((32,), dtype=mx.float32), sample_rate=44100)
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1
        assert f.getnframes() == 32
