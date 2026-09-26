from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech.asr._adapter import ASROutput
from mlx_speech.asr._adapters.confucius4_r2t2 import (
    Confucius4R2T2Adapter,
    _prompt_text,
)
from mlx_speech.models.qwen3_asr import (
    Qwen3ASRFeatureExtractor,
    Qwen3ASRProcessor,
    _get_feat_extract_output_lengths,
)

HOP_LENGTH = 160


class FakeQwen3ASRTokenizer:
    """Character-level stand-in that keeps the audio specials single tokens."""

    audio_token = "<|audio_pad|>"
    audio_bos_token = "<|audio_start|>"
    audio_eos_token = "<|audio_end|>"
    audio_token_id = 151676
    audio_bos_token_id = 151669
    audio_eos_token_id = 151670

    _specials = {
        "<|im_start|>": 151644,
        "<|im_end|>": 151645,
        audio_bos_token: audio_bos_token_id,
        audio_eos_token: audio_eos_token_id,
        audio_token: audio_token_id,
        "<asr_text>": 151704,
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        ids: list[int] = []
        index = 0
        ordered = sorted(self._specials, key=len, reverse=True)
        while index < len(text):
            for token in ordered:
                if text.startswith(token, index):
                    ids.append(self._specials[token])
                    index += len(token)
                    break
            else:
                ids.append(1000 + ord(text[index]))
                index += 1
        return ids

    def decode(self, token_ids, *, skip_special_tokens: bool = False) -> str:
        reverse = {value: key for key, value in self._specials.items()}
        parts: list[str] = []
        for token_id in token_ids:
            value = int(token_id)
            if value in reverse:
                if not skip_special_tokens:
                    parts.append(reverse[value])
            else:
                parts.append(chr(value - 1000))
        return "".join(parts)


def _processor() -> Qwen3ASRProcessor:
    return Qwen3ASRProcessor(
        config=SimpleNamespace(support_languages=("Chinese", "English")),
        tokenizer=FakeQwen3ASRTokenizer(),
        feature_extractor=Qwen3ASRFeatureExtractor(),
    )


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(processor=_processor())


def _scripted(*outputs: str):
    pending = list(outputs)
    calls: list[tuple[str, int, int]] = []

    def step(prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
        calls.append((prefix, int(audio.shape[0]), int(max_new_tokens)))
        assert pending, "step decoder called more times than scripted"
        return pending.pop(0)

    return step, calls


def test_stream_session_commits_scripted_text_and_finalize_returns_output():
    step, calls = _scripted("hello world")
    adapter = Confucius4R2T2Adapter(_runtime(), step_decoder=step)
    session = adapter.stream_session(
        language="English", chunk_ms=80, lookahead_ms=0, unfixed_token_num=0
    )

    update = session.feed(np.zeros(1280, dtype=np.float32))

    assert update.hypothesis == "hello world"
    assert update.committed == "hello world"
    assert update.delta == "hello world"
    assert calls == [("", 1280, 1)]

    result = session.finalize()

    assert isinstance(result, ASROutput)
    assert result.text == "hello world"
    assert result.language == "English"
    assert calls == [("", 1280, 1)]


def test_stream_session_rolls_back_one_token_into_the_next_prefix():
    step, calls = _scripted("abcd", "next")
    adapter = Confucius4R2T2Adapter(_runtime(), step_decoder=step)
    session = adapter.stream_session(
        language="English", chunk_ms=80, lookahead_ms=0, unfixed_token_num=1
    )

    first = session.feed(np.zeros(1280, dtype=np.float32))
    second = session.feed(np.zeros(1280, dtype=np.float32))

    assert first.committed == "abc"
    assert first.hypothesis == "abcd"
    # Prefix for chunk 2 is the held-back hypothesis minus one token.
    assert calls[1][0] == "abc"
    # The lock keeps the committed head and appends only the new tail by length,
    # so "abc" + "abcnex"[3:] rather than the replaced hypothesis "abcnext".
    assert second.delta == "nex"
    assert second.committed == "abcnex"
    assert session.language == "English"
    assert session.audio_samples == 2560
    assert session.chunk_id == 2


def test_stream_session_rejects_a_non_16khz_sample_rate():
    adapter = Confucius4R2T2Adapter(_runtime(), step_decoder=_scripted("x")[0])

    with pytest.raises(ValueError, match="16000"):
        adapter.stream_session(sample_rate=8000)


def test_prompt_text_appends_prefix_after_the_forced_language_suffix():
    processor = _processor()

    prompt = _prompt_text(
        processor,
        prompt_suffix="你",
        audio_length=2,
        context="",
        language="Chinese",
    )

    assert (
        prompt
        == processor.build_prompt(context="", audio_length=2, language="Chinese").prompt
        + "你"
    )
    assert prompt.startswith(
        "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>"
    )
    assert prompt.endswith("language Chinese<asr_text>你")
    assert prompt.count("<|audio_pad|>") == 2


def test_prompt_text_without_forced_language_keeps_the_bare_assistant_tail():
    processor = _processor()

    prompt = _prompt_text(
        processor,
        prompt_suffix="hi",
        audio_length=1,
        context="",
        language=None,
    )

    assert prompt.endswith("<|im_start|>assistant\nhi")
    assert "language " not in prompt
    assert "<asr_text>" not in prompt


def test_pad_count_follows_the_feature_formula_not_samples_per_1280():
    for samples, audio_tokens in ((1280, 1), (2560, 2), (5120, 4)):
        frames = samples // HOP_LENGTH
        assert _get_feat_extract_output_lengths(frames) == audio_tokens
