from __future__ import annotations

import argparse
import wave

import mlx.core as mx
import pytest

import mlx_speech.tts as tts
from mlx_speech.tts import TTSOutput
from mlx_speech.tts.generate import add_tts_args, tts_main


class _BatchModel:
    def __init__(self) -> None:
        self.calls = []

    def generate(self, text=None, **kwargs):
        self.calls.append((text, kwargs))
        return TTSOutput(mx.ones((8,)), 48_000)


class _StreamingModel(_BatchModel):
    def __init__(self) -> None:
        super().__init__()
        self.stream_calls = []

    def generate_stream(self, text=None, **kwargs):
        self.stream_calls.append((text, kwargs))
        yield TTSOutput(mx.ones((3,)), 48_000)
        yield TTSOutput(mx.ones((5,)) * 0.5, 48_000)


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_tts_args(parser)
    return parser.parse_args(argv)


def test_cli_allows_speaker_only_reference_and_forwards_artifact_selector(
    monkeypatch,
    tmp_path,
) -> None:
    model = _BatchModel()
    loads = []

    def load(name, **kwargs):
        loads.append((name, kwargs))
        return model

    monkeypatch.setattr(tts, "load", load)
    output = tmp_path / "speaker.wav"
    tts_main(
        _args(
            "--model",
            "appautomaton/dots-tts-mlx",
            "--artifact-subdir",
            "mf/mlx-int8",
            "--text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--output",
            str(output),
        )
    )
    assert loads == [
        (
            "appautomaton/dots-tts-mlx",
            {"artifact_subdir": "mf/mlx-int8", "codec_path_or_repo": None},
        )
    ]
    assert model.calls == [("hello", {"reference_audio": "ref.wav"})]
    assert output.is_file()


def test_cli_streams_without_collecting_chunks(monkeypatch, tmp_path) -> None:
    model = _StreamingModel()
    monkeypatch.setattr(tts, "load", lambda *args, **kwargs: model)
    output = tmp_path / "stream.wav"
    tts_main(
        _args(
            "--model",
            "dots-tts-soar",
            "--text",
            "hello",
            "--stream",
            "--stream-chunk-patches",
            "3",
            "--output",
            str(output),
        )
    )
    assert model.calls == []
    assert model.stream_calls == [("hello", {"stream_chunk_patches": 3})]
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getframerate() == 48_000
        assert wav_file.getnframes() == 8


def test_cli_rejects_streaming_for_non_streaming_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts, "load", lambda *args, **kwargs: _BatchModel())
    with pytest.raises(ValueError, match="does not support waveform streaming"):
        tts_main(
            _args(
                "--model",
                "fish-s2-pro",
                "--text",
                "hello",
                "--stream",
                "--output",
                str(tmp_path / "unused.wav"),
            )
        )
