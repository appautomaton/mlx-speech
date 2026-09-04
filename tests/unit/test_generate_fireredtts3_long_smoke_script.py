from __future__ import annotations

import re

import mlx.core as mx
import numpy as np

from mlx_speech.tts import TTSOutput
from scripts.generate.fireredtts3_long_smoke import (
    DEFAULT_OUTPUT,
    DEFAULT_REFERENCE_AUDIO,
    DEFAULT_TEXT,
    _build_parser,
    clean_text,
    cross_fade_waveforms,
    run,
    split_text,
)


def test_default_smoke_text_is_an_extended_technical_explanation() -> None:
    chinese_characters = re.findall(r"[\u4e00-\u9fff]", DEFAULT_TEXT)

    assert 500 <= len(chinese_characters) <= 530
    for module_name in ("RedAE", "CAM++", "Qwen3", "PatchEncoder", "DiT"):
        assert module_name in DEFAULT_TEXT


def test_parser_defaults_reuse_the_local_golden_fixture() -> None:
    args = _build_parser().parse_args([])

    assert args.reference_audio == DEFAULT_REFERENCE_AUDIO
    assert args.output == DEFAULT_OUTPUT
    assert args.language == "Chinese"
    assert args.seed == 1234
    assert args.guidance_scale == 2.0
    assert args.flow_steps == 10
    assert args.stop_threshold == 0.5
    assert args.max_audio_patches == 400
    assert args.segments_file is None
    assert args.no_split is False
    assert args.cross_fade_ms == 50.0


def test_default_text_uses_the_official_soft_split_boundaries() -> None:
    segments = split_text(DEFAULT_TEXT, language="Chinese")

    assert [len(segment) for segment in segments] == [73, 97, 109, 62, 95, 86, 93]
    assert "".join(segments) == DEFAULT_TEXT


def test_caller_text_helpers_clean_split_and_cross_fade() -> None:
    source = "# 标题\n- [链接](https://example.com) 😀\u200b **内容**"
    assert clean_text(source) == "标题 链接 内容"
    assert split_text("版本一点五写作1.5。下一句。", language="Chinese") == [
        "版本一点五写作1.5。下一句。"
    ]

    waveform = cross_fade_waveforms(
        [mx.ones((4,), dtype=mx.float32), mx.full((4,), 3.0, dtype=mx.float32)],
        sample_rate=1_000,
        cross_fade_ms=2.0,
    )
    np.testing.assert_array_equal(waveform, [1.0, 1.0, 1.0, 3.0, 3.0, 3.0])


def test_long_smoke_orchestrates_multiple_runtime_calls(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    reference_audio = tmp_path / "reference.wav"
    reference_audio.touch()
    output = tmp_path / "output.wav"
    first = "甲" * 61 + "。"
    second = "乙" * 61 + "。"

    class _Model:
        tokenizer = type("Tokenizer", (), {"count_tokens": staticmethod(len)})()

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, text: str, **kwargs) -> TTSOutput:
            self.calls.append(text)
            return TTSOutput(mx.ones((4,), dtype=mx.float32), 24_000)

    model = _Model()
    monkeypatch.setattr(
        "scripts.generate.fireredtts3_long_smoke.tts.load",
        lambda path: model,
    )
    args = _build_parser().parse_args(
        [
            "--model-dir",
            str(model_dir),
            "--reference-audio",
            str(reference_audio),
            "--output",
            str(output),
            "--text",
            first + second,
            "--cross-fade-ms",
            "0",
        ]
    )

    assert run(args) == output
    assert model.calls == [first, second]
    assert output.is_file()


def test_long_smoke_accepts_caller_prepared_utterances(monkeypatch, tmp_path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    reference_audio = tmp_path / "reference.wav"
    reference_audio.touch()
    segments_file = tmp_path / "segments.txt"
    segments_file.write_text("第一段。\n\n第二段。\n", encoding="utf-8")
    output = tmp_path / "output.wav"

    class _Model:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, text: str, **kwargs) -> TTSOutput:
            self.calls.append(text)
            return TTSOutput(mx.ones((4,), dtype=mx.float32), 24_000)

    model = _Model()
    monkeypatch.setattr(
        "scripts.generate.fireredtts3_long_smoke.tts.load",
        lambda path: model,
    )
    args = _build_parser().parse_args(
        [
            "--model-dir",
            str(model_dir),
            "--reference-audio",
            str(reference_audio),
            "--segments-file",
            str(segments_file),
            "--output",
            str(output),
            "--cross-fade-ms",
            "0",
        ]
    )

    assert run(args) == output
    assert model.calls == ["第一段。", "第二段。"]
