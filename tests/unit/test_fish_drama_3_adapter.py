"""Drama 3 adapter: reference handling as upstream fish-speech, and registry routing."""

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.generation.fish_drama_3 import Drama3Chunk, Drama3Reference
from mlx_speech.tts._adapters.fish_drama_3 import FishDrama3Adapter
from mlx_speech.tts._registry import _resolve_tts_family


class FakeRuntime:
    sample_rate = 44_100

    def __init__(self):
        self.encoded, self.calls = [], []

    def encode_reference(self, audio, text, *, speaker=None):
        self.encoded.append((str(audio), text, speaker))
        return Drama3Reference(codes=np.zeros((10, 1), np.int32), text=text)

    def synthesize(self, text, references, **kwargs):
        self.calls.append((text, references, kwargs))
        for size in (3, 2):
            yield Drama3Chunk(text, np.zeros((10, 1)), mx.ones((size,)), "eos")


def test_parallel_reference_lists_are_encoded_with_their_speakers():
    runtime = FakeRuntime()

    output = FishDrama3Adapter(runtime).generate(
        "<|speaker:0|> Are you coming? <|speaker:1|> Give me a minute.",
        reference_audio=["anna.wav", "ben.wav"],
        reference_text=["Anna's transcript.", "Ben's transcript."],
        reference_speaker=[0, 1],
        top_k=5,
    )

    assert runtime.encoded == [
        ("anna.wav", "Anna's transcript.", 0),
        ("ben.wav", "Ben's transcript.", 1),
    ]
    _, references, kwargs = runtime.calls[0]
    assert len(references) == 2 and kwargs["top_k"] == 5
    assert kwargs["max_new_tokens"] is None and kwargs["duration_seconds"] is None
    assert output.waveform.shape == (5,) and output.sample_rate == 44_100


def test_duration_reaches_the_runtime():
    runtime = FakeRuntime()

    FishDrama3Adapter(runtime).generate("你好。", duration_seconds=1.5)

    assert runtime.calls[0][2]["duration_seconds"] == 1.5


def test_a_prepared_reference_is_reused_without_encoding_again():
    runtime = FakeRuntime()
    adapter = FishDrama3Adapter(runtime)
    voice = adapter.prepare_reference("anna.wav", reference_text="Hi.")

    adapter.generate("Hello.", reference_audio=voice)
    adapter.generate("Again.", reference_audio=[voice])

    assert len(runtime.encoded) == 1
    assert [refs for _, refs, _ in runtime.calls] == [[voice], [voice]]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"reference_text": "orphan"}, ValueError, "need reference_audio"),
        ({"reference_audio": "a.wav"}, ValueError, "needs its transcript"),
        (
            {"reference_audio": ["a.wav", "b.wav"], "reference_text": ["one"]},
            ValueError,
            "one entry per reference",
        ),
        (
            {"reference_audio": ["a.wav", "b.wav"], "reference_text": "both"},
            ValueError,
            "must be a list",
        ),
        ({"reference_audio": mx.zeros(10), "reference_text": "x"}, TypeError, "path"),
    ],
)
def test_rejects_inconsistent_references(kwargs, error, match):
    with pytest.raises(error, match=match):
        FishDrama3Adapter(FakeRuntime()).generate("Hello.", **kwargs)


def test_a_prepared_reference_rejects_a_second_transcript():
    voice = Drama3Reference(codes=np.zeros((10, 1), np.int32), text="Hi.")

    with pytest.raises(ValueError, match="already carries"):
        FishDrama3Adapter(FakeRuntime()).generate(
            "Hello.", reference_audio=voice, reference_text="Hi."
        )


def test_registry_routes_fish_omni_checkpoints_to_drama_3(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "fish_omni"}))

    assert _resolve_tts_family(tmp_path) == "fish_drama_3"
