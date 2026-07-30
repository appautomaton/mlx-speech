from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_speech.tts._adapters.dots_tts import DotsTTSAdapter
from mlx_speech.tts._registry import _resolve_tts_family


class _Generator:
    def __init__(self):
        self.kwargs = None

    def synthesize(self, text, **kwargs):
        self.kwargs = {"text": text, **kwargs}
        return SimpleNamespace(waveform=mx.ones((8,)), sample_rate=48_000)


def test_adapter_keeps_model_controls_in_backend_kwargs() -> None:
    generator = _Generator()
    adapter = DotsTTSAdapter(generator)
    output = adapter.generate(
        "hello",
        reference_audio=mx.ones((16,)),
        reference_text="reference",
        max_new_tokens=7,
        reference_sample_rate=24_000,
        solver_steps=3,
        guidance_scale=1.4,
        speaker_scale=1.6,
        language="en",
        seed=9,
        eos_threshold=0.7,
    )
    assert output.sample_rate == 48_000
    assert generator.kwargs["max_audio_patches"] == 7
    assert generator.kwargs["reference_sample_rate"] == 24_000
    assert generator.kwargs["solver_steps"] == 3
    assert generator.kwargs["seed"] == 9


def test_adapter_accepts_explicit_patch_budget_and_rejects_conflicts() -> None:
    generator = _Generator()
    adapter = DotsTTSAdapter(generator)
    adapter.generate("hello", max_audio_patches=5)
    assert generator.kwargs["max_audio_patches"] == 5
    with pytest.raises(ValueError, match="must match"):
        adapter.generate("hello", max_new_tokens=4, max_audio_patches=5)


def test_registry_detects_dots_tts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"dots_tts"}')
    assert _resolve_tts_family(tmp_path) == "dots_tts"
