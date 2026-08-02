from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.tts import TTSOutput
from mlx_speech.tts._adapters.dots_tts import DotsTTSAdapter
from mlx_speech.tts._registry import _resolve_tts_family


class _Generator:
    sample_rate = 48_000

    def __init__(self):
        self.kwargs = None
        self.synthesize_calls = 0
        self.stream_calls = 0

    def synthesize(self, text, **kwargs):
        self.synthesize_calls += 1
        self.kwargs = {"text": text, **kwargs}
        return SimpleNamespace(
            waveform=mx.arange(6, dtype=mx.float32),
            num_patches=3,
        )

    def synthesize_stream(self, text, **kwargs):
        self.stream_calls += 1
        self.kwargs = {"text": text, **kwargs}
        yield SimpleNamespace(waveform=mx.ones((3,)), num_patches=1)
        yield SimpleNamespace(waveform=mx.ones((5,)) * 2, num_patches=2)


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
    np.testing.assert_array_equal(
        output.waveform,
        mx.arange(6, dtype=mx.float32),
    )
    assert generator.synthesize_calls == 1
    assert generator.stream_calls == 0
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


def test_adapter_streams_unified_outputs_and_keeps_patch_metadata_private() -> None:
    generator = _Generator()
    adapter = DotsTTSAdapter(generator)
    chunks = list(adapter.generate_stream("hello", stream_chunk_patches=3))

    assert all(isinstance(chunk, TTSOutput) for chunk in chunks)
    assert [int(chunk.waveform.size) for chunk in chunks] == [3, 5]
    assert all(chunk.sample_rate == 48_000 for chunk in chunks)
    assert all(not hasattr(chunk, "num_patches") for chunk in chunks)
    assert generator.synthesize_calls == 0
    assert generator.stream_calls == 1
    assert generator.kwargs["stream_chunk_patches"] == 3
    with pytest.raises(ValueError, match="positive integer"):
        list(adapter.generate_stream("hello", stream_chunk_patches=0))


def test_registry_detects_dots_tts(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"dots_tts"}')
    assert _resolve_tts_family(tmp_path) == "dots_tts"
