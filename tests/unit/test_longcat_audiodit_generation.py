from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from mlx_speech import generation as generation_exports
from mlx_speech.generation.longcat_audiodit import (
    LongCatSynthesisOutput,
    build_longcat_full_text,
    parse_longcat_batch_manifest_line,
    synthesize_longcat_audiodit,
)
import pytest


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode_text(self, texts: list[str]) -> dict[str, list[list[int]]]:
        self.calls.append(texts)
        return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}


class _FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            sampling_rate=24000, latent_hop=2048, max_wav_duration=60
        )
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            waveform=mx.zeros((1, 16), dtype=mx.float32),
            latent=mx.zeros((1, 4, 2), dtype=mx.float32),
        )


def test_build_longcat_full_text_matches_single_and_batch_rules() -> None:
    assert (
        build_longcat_full_text("Target", "Prompt", batch_mode=False) == "prompt target"
    )
    assert (
        build_longcat_full_text("Target", "Prompt.", batch_mode=True)
        == "prompt. target"
    )
    assert (
        build_longcat_full_text("Target", "Prompt", batch_mode=True) == "prompttarget"
    )


def test_parse_batch_manifest_line_matches_upstream_layout() -> None:
    item = parse_longcat_batch_manifest_line(
        "utt-1|prompt text|prompt.wav|generated text",
        line_number=3,
    )
    assert item.uid == "utt-1"
    assert item.prompt_text == "prompt text"
    assert item.prompt_wav_path == Path("prompt.wav")
    assert item.gen_text == "generated text"


def test_synthesize_longcat_audiodit_returns_result_dataclass() -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer()

    result = synthesize_longcat_audiodit(
        model=model,
        tokenizer=tokenizer,
        text="  Hello  “World”  ",
        prompt_text="Prompt line.",
        prompt_audio=None,
        nfe=12,
        guidance_method="cfg",
        guidance_strength=3.5,
    )

    assert isinstance(result, LongCatSynthesisOutput)
    assert result.sample_rate == 24000
    assert result.latent_frames == 2
    assert tokenizer.calls == [["prompt line.  hello world "]]
    assert model.calls[0]["steps"] == 12
    assert model.calls[0]["cfg_strength"] == 3.5


def test_generation_package_exports_longcat_surface() -> None:
    assert generation_exports.LongCatSynthesisOutput is LongCatSynthesisOutput
    assert generation_exports.synthesize_longcat_audiodit is synthesize_longcat_audiodit


def test_synthesize_longcat_audiodit_rejects_prompt_audio_without_prompt_text() -> None:
    model = _FakeModel()
    tokenizer = _FakeTokenizer()

    with pytest.raises(ValueError, match="prompt_text"):
        synthesize_longcat_audiodit(
            model=model,
            tokenizer=tokenizer,
            text="hello",
            prompt_text=None,
            prompt_audio=mx.zeros((1, 1, 8), dtype=mx.float32),
        )
