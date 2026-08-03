from __future__ import annotations

import wave

import mlx.core as mx

from mlx_speech.tts import TTSOutput
from scripts.generate import dots_tts as generate_script


class _Model:
    def __init__(self) -> None:
        self.batch_calls = []
        self.stream_calls = []

    def generate(self, text, **kwargs):
        self.batch_calls.append((text, kwargs))
        return TTSOutput(mx.ones((6,)), 48_000)

    def generate_stream(self, text, **kwargs):
        self.stream_calls.append((text, kwargs))
        yield TTSOutput(mx.ones((2,)), 48_000)
        yield TTSOutput(mx.ones((4,)), 48_000)


def test_dots_script_maps_full_batch_surface(monkeypatch, tmp_path) -> None:
    model = _Model()
    loads = []

    def load(name, **kwargs):
        loads.append((name, kwargs))
        return model

    monkeypatch.setattr(generate_script.tts, "load", load)
    output = tmp_path / "batch.wav"
    args = generate_script.parse_args(
        [
            "--model",
            "appautomaton/dots-tts-mlx",
            "--artifact-subdir",
            "soar/mlx-base",
            "--text",
            "hello",
            "--reference-audio",
            "ref.wav",
            "--reference-text",
            "reference",
            "--max-audio-patches",
            "17",
            "--solver-steps",
            "3",
            "--guidance-scale",
            "1.4",
            "--speaker-scale",
            "1.6",
            "--language",
            "en",
            "--seed",
            "9",
            "--eos-threshold",
            "0.7",
            "--template",
            "tts_interleave",
            "--output",
            str(output),
        ]
    )
    generate_script.run(args)
    assert loads == [
        (
            "appautomaton/dots-tts-mlx",
            {"artifact_subdir": "soar/mlx-base"},
        )
    ]
    _, kwargs = model.batch_calls[0]
    assert kwargs == {
        "reference_audio": "ref.wav",
        "reference_text": "reference",
        "max_audio_patches": 17,
        "solver_steps": 3,
        "guidance_scale": 1.4,
        "speaker_scale": 1.6,
        "language": "en",
        "seed": 9,
        "eos_threshold": 0.7,
        "template": "tts_interleave",
    }
    assert output.is_file()


def test_dots_script_streams_speaker_only(monkeypatch, tmp_path) -> None:
    model = _Model()
    monkeypatch.setattr(generate_script.tts, "load", lambda *args, **kwargs: model)
    output = tmp_path / "stream.wav"
    generate_script.run(
        generate_script.parse_args(
            [
                "--text",
                "hello",
                "--reference-audio",
                "ref.wav",
                "--stream",
                "--stream-chunk-patches",
                "2",
                "--output",
                str(output),
            ]
        )
    )
    assert model.batch_calls == []
    _, kwargs = model.stream_calls[0]
    assert kwargs["reference_audio"] == "ref.wav"
    assert kwargs["stream_chunk_patches"] == 2
    with wave.open(str(output), "rb") as wav_file:
        assert wav_file.getnframes() == 6
