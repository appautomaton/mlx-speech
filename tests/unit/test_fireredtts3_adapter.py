from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_speech.tts as tts
from mlx_speech.tts import TTSOutput
from mlx_speech.tts._adapters.fireredtts3 import FireRedTTS3Adapter
from mlx_speech.tts._registry import _resolve_tts_family
from mlx_speech.tts.generate import add_tts_args, tts_main


class _Core:
    config = SimpleNamespace(patch_size=4)

    def __init__(self) -> None:
        self.kwargs = None
        self.calls = []

    def generate(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)
        prompt = kwargs["prompt_latents"]
        generated = mx.ones((1, 4, 2), dtype=mx.float32)
        return SimpleNamespace(latents=mx.concatenate((prompt, generated), axis=1))


class _RedAE:
    sample_rate = 24_000
    downsample_rate = 2

    def __init__(self) -> None:
        self.encode_calls = 0
        self.decode_calls = 0

    def pad_audio(self, audio, *, multiple=None):
        padding = (-int(audio.shape[-1])) % int(multiple)
        return mx.pad(audio, ((0, 0), (padding, 0)))

    def encode(self, audio):
        self.encode_calls += 1
        return audio.reshape(1, -1, 2)

    def decode(self, latents):
        self.decode_calls += 1
        return latents.reshape(1, -1)


class _Speaker:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, audio, *, sample_rate):
        assert sample_rate == 24_000
        self.calls += 1
        return mx.ones((1, 6), dtype=mx.float32)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, **kwargs):
        self.calls.append(kwargs)
        return [1, 2, 3]


def _adapter() -> tuple[FireRedTTS3Adapter, _Core]:
    core = _Core()
    return (
        FireRedTTS3Adapter(
            core=core,
            redae=_RedAE(),
            speaker=_Speaker(),
            tokenizer=_Tokenizer(),
        ),
        core,
    )


def test_adapter_prepares_reference_and_returns_trimmed_24khz_waveform() -> None:
    adapter, core = _adapter()
    output = adapter.generate(
        "目标文本",
        reference_audio=mx.linspace(-0.1, 0.1, 5),
        reference_sample_rate=24_000,
        reference_text="参考文本",
        language="Chinese",
        seed=7,
        guidance_scale=1.5,
        flow_steps=3,
        stop_threshold=0.6,
        max_audio_patches=9,
    )
    assert output.sample_rate == 24_000
    assert output.waveform.shape == (8,)
    assert core.kwargs["seed"] == 7
    assert core.kwargs["guidance_scale"] == 1.5
    assert core.kwargs["flow_steps"] == 3
    assert core.kwargs["stop_threshold"] == 0.6
    assert core.kwargs["max_generated_patches"] == 9
    np.testing.assert_array_equal(core.kwargs["text_tokens"], [[1, 2, 3]])
    assert adapter.tokenizer.calls == [
        {
            "language": "Chinese",
            "reference_text": "参考文本",
            "text": "目标文本",
        }
    ]


def test_adapter_treats_multiple_sentences_as_one_prepared_utterance() -> None:
    adapter, core = _adapter()
    first = "甲" * 61 + "。"
    second = "乙" * 61 + "。"

    output = adapter.generate(
        first + second,
        reference_audio=mx.linspace(-0.1, 0.1, 5),
        reference_sample_rate=24_000,
        reference_text="参考文本",
        language="Chinese",
        seed=7,
    )

    assert output.waveform.shape == (8,)
    assert [call["seed"] for call in core.calls] == [7]
    assert [call["max_generated_patches"] for call in core.calls] == [400]
    assert [call["text"] for call in adapter.tokenizer.calls] == [first + second]
    assert adapter.redae.encode_calls == 1
    assert adapter.redae.decode_calls == 1
    assert adapter.speaker.calls == 1


def test_adapter_uses_a_predictable_english_language_default() -> None:
    adapter, _ = _adapter()

    adapter.generate(
        "Prepared utterance.",
        reference_audio=mx.ones((8,)),
        reference_text="Reference transcript.",
    )

    assert adapter.tokenizer.calls[0]["language"] == "English"


def test_adapter_rejects_invalid_cloning_inputs_and_patch_conflict() -> None:
    adapter, _ = _adapter()
    with pytest.raises(ValueError, match="requires reference_audio"):
        adapter.generate("text", reference_text="transcript")
    with pytest.raises(ValueError, match="must match"):
        adapter.generate(
            "目标文本",
            reference_audio=mx.ones((8,)),
            reference_text="参考文本",
            language="Chinese",
            max_new_tokens=2,
            max_audio_patches=3,
        )


def test_registry_detects_fireredtts3_base(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(
        '{"model_type":"fireredtts3_base"}', encoding="utf-8"
    )
    assert _resolve_tts_family(tmp_path) == "fireredtts3"


def test_cli_exposes_firered_generation_controls() -> None:
    parser = argparse.ArgumentParser()
    add_tts_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "local",
            "--text",
            "你好",
            "--reference-audio",
            "ref.wav",
            "--reference-text",
            "参考",
            "--language",
            "Chinese",
            "--seed",
            "9",
            "--guidance-scale",
            "2.5",
            "--flow-steps",
            "12",
            "--stop-threshold",
            "0.4",
            "--max-audio-patches",
            "20",
        ]
    )
    assert args.language == "Chinese"
    assert args.seed == 9
    assert args.guidance_scale == 2.5
    assert args.flow_steps == 12
    assert args.stop_threshold == 0.4
    assert args.max_audio_patches == 20


def test_cli_forwards_firered_generation_controls(monkeypatch, tmp_path: Path) -> None:
    class _Model:
        kwargs = None

        def generate(self, text, **kwargs):
            self.kwargs = {"text": text, **kwargs}
            return TTSOutput(mx.ones((8,)), 24_000)

    model = _Model()
    monkeypatch.setattr(tts, "load", lambda *args, **kwargs: model)
    parser = argparse.ArgumentParser()
    add_tts_args(parser)
    args = parser.parse_args(
        [
            "--model",
            "local",
            "--text",
            "你好",
            "--reference-audio",
            "ref.wav",
            "--reference-text",
            "参考",
            "--language",
            "Chinese",
            "--seed",
            "9",
            "--guidance-scale",
            "2.5",
            "--flow-steps",
            "12",
            "--stop-threshold",
            "0.4",
            "--max-audio-patches",
            "20",
            "--output",
            str(tmp_path / "out.wav"),
        ]
    )
    tts_main(args)
    assert model.kwargs == {
        "text": "你好",
        "reference_audio": "ref.wav",
        "reference_text": "参考",
        "max_audio_patches": 20,
        "language": "Chinese",
        "seed": 9,
        "guidance_scale": 2.5,
        "flow_steps": 12,
        "stop_threshold": 0.4,
    }
