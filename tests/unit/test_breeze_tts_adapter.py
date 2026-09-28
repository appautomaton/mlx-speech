"""Breeze TTS uses the shared generation controls through the library API."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.generation.breeze_tts import BreezeAudio
from mlx_speech.tts import StreamingTTSModel, TTSOutput
from mlx_speech.tts._adapters.breeze_tts import BreezeTTSAdapter
from mlx_speech.tts._registry import _resolve_tts_family


class _Adapter(BreezeTTSAdapter):
    def __init__(self) -> None:
        super().__init__(object(), object(), object())


def test_adapter_maps_direction_and_reference_controls(monkeypatch) -> None:
    captured = {}

    def generate(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return BreezeAudio(mx.ones((4,), dtype=mx.float32), 24_000, 2)

    monkeypatch.setattr("mlx_speech.tts._adapters.breeze_tts.generate_breeze", generate)
    adapter = _Adapter()
    reference = mx.ones((8, 2), dtype=mx.float32)
    output = adapter.generate(
        " 你好 ",
        reference_audio=reference,
        reference_text=" hello ",
        reference_sample_rate=48_000,
        max_new_tokens=12,
        guidance_scale=4.0,
        instruction="说得活泼一点",
        seed=42,
        ignored="drop me",
    )

    assert isinstance(adapter, StreamingTTSModel)
    assert isinstance(output, TTSOutput)
    assert output.sample_rate == 24_000
    assert int(output.waveform.size) == 4
    assert captured["kwargs"]["speaker"] == "S0"
    assert captured["kwargs"]["instruction"] == "说得活泼一点"
    assert captured["kwargs"]["cfg_scale"] == 4.0
    assert captured["kwargs"]["max_frames"] == 12
    assert captured["kwargs"]["seed"] == 42
    assert captured["kwargs"]["ref_text"] == " hello "
    assert captured["kwargs"]["ref_audio"].shape == (4,)


def test_adapter_rejects_conflicting_controls() -> None:
    adapter = _Adapter()
    with pytest.raises(ValueError, match="cfg_scale and guidance_scale"):
        adapter.generate("hello", cfg_scale=2.0, guidance_scale=4.0)
    with pytest.raises(ValueError, match="max_new_tokens and max_frames"):
        adapter.generate("hello", max_new_tokens=4, max_frames=8)
    with pytest.raises(ValueError, match="requires text"):
        adapter.generate("   ")
    with pytest.raises(ValueError, match="positive integer"):
        list(adapter.generate_stream("hello", stream_chunk_patches=0))


def test_adapter_streams_without_calling_the_offline_decoder(monkeypatch) -> None:
    calls = {"offline": 0}

    def generate(*args, **kwargs):
        del args, kwargs
        calls["offline"] += 1
        raise AssertionError("streaming should not decode the full utterance first")

    def generate_stream(*args, **kwargs):
        del args
        assert kwargs["chunk_frames"] == 3
        yield mx.ones((3,), dtype=mx.float32)
        yield mx.ones((1,), dtype=mx.float32)

    monkeypatch.setattr("mlx_speech.tts._adapters.breeze_tts.generate_breeze", generate)
    monkeypatch.setattr(
        "mlx_speech.tts._adapters.breeze_tts.generate_breeze_stream", generate_stream
    )
    chunks = list(_Adapter().generate_stream("hello", stream_chunk_patches=3))
    assert calls["offline"] == 0
    assert [int(chunk.waveform.size) for chunk in chunks] == [3, 1]
    assert all(chunk.sample_rate == 24_000 for chunk in chunks)


def test_registry_detects_the_mlx_package_and_model_type(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "config.json").write_text(
        '{"model_type":"breeze","mlx_speech":{"family":"breeze_tts"}}',
        encoding="utf-8",
    )
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "config.json").write_text('{"model_type":"breeze"}', encoding="utf-8")
    assert _resolve_tts_family(package) == "breeze_tts"
    assert _resolve_tts_family(plain) == "breeze_tts"
