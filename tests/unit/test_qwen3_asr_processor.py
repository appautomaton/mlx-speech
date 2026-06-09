from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech.models.qwen3_asr import (
    Qwen3ASRFeatureExtractor,
    Qwen3ASRProcessor,
    Qwen3ASRTokenizer,
    parse_asr_output,
)


QWEN_DIR = Path("models/qwen3_asr_1_7b/original")
TOKENIZER_FILES = (
    QWEN_DIR / "config.json",
    QWEN_DIR / "tokenizer_config.json",
    QWEN_DIR / "vocab.json",
    QWEN_DIR / "merges.txt",
)


class FakeQwen3ASRTokenizer:
    audio_token = "<|audio_pad|>"
    audio_bos_token = "<|audio_start|>"
    audio_eos_token = "<|audio_end|>"
    audio_token_id = 10
    audio_bos_token_id = 11
    audio_eos_token_id = 12

    _specials = {
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        audio_token: audio_token_id,
        audio_bos_token: audio_bos_token_id,
        audio_eos_token: audio_eos_token_id,
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids: list[int] = []
        index = 0
        special_tokens = sorted(self._specials, key=len, reverse=True)
        while index < len(text):
            for token in special_tokens:
                if text.startswith(token, index):
                    ids.append(self._specials[token])
                    index += len(token)
                    break
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return ids


def _processor() -> Qwen3ASRProcessor:
    return Qwen3ASRProcessor(
        config=SimpleNamespace(support_languages=("Chinese", "English")),
        tokenizer=FakeQwen3ASRTokenizer(),
        feature_extractor=Qwen3ASRFeatureExtractor(),
    )


def _deterministic_waveform(samples: int) -> np.ndarray:
    t = np.arange(samples, dtype=np.float32) / 16000.0
    return (0.1 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.mark.parametrize("language", [None, "", "auto", " AUTO "])
def test_qwen3_asr_auto_language_prompt_has_no_forced_suffix(language):
    processor = _processor()

    prompt = processor.build_prompt(audio_length=2, language=language)

    assert prompt.language is None
    assert prompt.prompt.endswith("<|im_start|>assistant\n")
    assert "language " not in prompt.prompt
    assert "<asr_text>" not in prompt.prompt
    assert prompt.input_ids.count(processor.tokenizer.audio_token_id) == 2
    assert prompt.input_ids.count(processor.tokenizer.audio_bos_token_id) == 1
    assert prompt.input_ids.count(processor.tokenizer.audio_eos_token_id) == 1


def test_qwen3_asr_forced_language_prompt_appends_suffix():
    processor = _processor()

    prompt = processor.build_prompt(audio_length=1, language="english")

    assert prompt.language == "English"
    assert prompt.prompt.endswith("language English<asr_text>")
    assert prompt.input_ids.count(processor.tokenizer.audio_token_id) == 1


def test_qwen3_asr_processor_rejects_unsupported_language():
    processor = _processor()

    with pytest.raises(ValueError, match="Unsupported language"):
        processor.build_prompt(audio_length=1, language="Korean")


def test_qwen3_asr_placeholder_count_matches_computed_audio_lengths_for_batch():
    processor = _processor()
    short = _deterministic_waveform(1600)
    long = _deterministic_waveform(3200)

    output = processor(
        [short, long],
        context=["", "domain words"],
        language=[None, "Chinese"],
        sample_rate=16000,
    )

    assert output.audio_lengths.tolist() == [2, 3]
    assert [ids.count(processor.tokenizer.audio_token_id) for ids in output.input_ids] == [2, 3]
    assert output.languages == (None, "Chinese")
    assert output.prompts[1].endswith("language Chinese<asr_text>")


def test_qwen3_asr_batch_prompt_rejects_mismatched_lengths():
    processor = _processor()

    with pytest.raises(ValueError, match="contexts length"):
        processor.build_batch_prompts(audio_lengths=[1, 2], contexts=["only one"])


def test_qwen3_asr_processor_uses_real_tokenizer_special_tokens_when_present():
    if not all(path.exists() for path in TOKENIZER_FILES):
        pytest.skip("Qwen3-ASR tokenizer assets not present")

    tokenizer = Qwen3ASRTokenizer.from_dir(QWEN_DIR)
    processor = Qwen3ASRProcessor(
        config=SimpleNamespace(support_languages=("Chinese", "English")),
        tokenizer=tokenizer,
        feature_extractor=Qwen3ASRFeatureExtractor(),
    )

    prompt = processor.build_prompt(audio_length=3, language=None)

    assert prompt.input_ids.count(151676) == 3
    assert prompt.input_ids.count(151669) == 1
    assert prompt.input_ids.count(151670) == 1


def test_qwen3_asr_parse_detected_language_output():
    assert parse_asr_output("language Chinese<asr_text>你好") == ("Chinese", "你好")
    assert parse_asr_output("language english\n<asr_text>hello") == ("English", "hello")


def test_qwen3_asr_parse_forced_language_output():
    assert parse_asr_output("hello world", user_language="english") == (
        "English",
        "hello world",
    )


def test_qwen3_asr_parse_empty_language_output():
    assert parse_asr_output("language None<asr_text>") == ("", "")
    assert parse_asr_output(None) == ("", "")
    assert parse_asr_output("") == ("", "")


def test_qwen3_asr_parse_no_language_tag_output():
    assert parse_asr_output("plain transcription") == ("", "plain transcription")


def test_qwen3_asr_parse_language_prefixed_output_without_asr_tag():
    raw = (
        "language EnglishHello from Qwen.\n\n"
        "language EnglishHello from Qwen again."
    )

    assert parse_asr_output(raw) == ("English", "Hello from Qwen.")


def test_qwen3_asr_parse_mixed_english_chinese_text_output():
    raw = "language Chinese<asr_text>今天 test 一下 Qwen ASR"

    assert parse_asr_output(raw) == ("Chinese", "今天 test 一下 Qwen ASR")
