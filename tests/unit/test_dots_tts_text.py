from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech.models.dots_tts.text import (
    DotsTTSTokenizer,
    build_generation_schedule,
    prepare_conditioned_text,
)


class _OracleBackend:
    _encodings = {
        "[EN]Oracle fixture sentence.": [
            58,
            953,
            60,
            48663,
            12507,
            11652,
            13,
        ],
        "[文本]": [58, 108704, 60],
        "[文本对应语音]": [58, 108704, 103124, 105761, 60],
        "[流式语音合成]": [58, 88653, 28330, 105761, 106726, 60],
    }

    def encode(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return SimpleNamespace(ids=self._encodings[text])


def _tokenizer() -> DotsTTSTokenizer:
    return DotsTTSTokenizer(
        backend=_OracleBackend(),
        audio_gen_start_id=151668,
        audio_gen_span_id=151669,
        audio_gen_end_id=151670,
        audio_comp_span_id=151666,
        text_cond_end_id=151671,
    )


def test_tts_schedule_matches_pinned_official_oracle() -> None:
    expected = np.load("tests/fixtures/dots_tts/soar/text_schedule.npz")
    schedule = build_generation_schedule(
        text="[EN]Oracle fixture sentence.",
        tokenizer=_tokenizer(),
        max_audio_patches=8,
    )
    assert schedule.token_ids == tuple(expected["tts_schedule"].tolist())
    assert schedule.audio_span_positions == tuple(range(16, 24))
    assert schedule.text_token_count == 7


def test_interleave_schedule_matches_pinned_official_oracle() -> None:
    expected = np.load("tests/fixtures/dots_tts/soar/text_schedule.npz")
    schedule = build_generation_schedule(
        text="[EN]Oracle fixture sentence.",
        tokenizer=_tokenizer(),
        max_audio_patches=24,
        template="tts_interleave",
    )
    assert schedule.token_ids == tuple(expected["interleave_schedule"].tolist())
    assert schedule.audio_patch_budget == 24
    assert schedule.interleave


def test_conditioned_text_tags_prompt_once_and_keeps_target_separate() -> None:
    prompt, target = prepare_conditioned_text(
        "New sentence.", language="en-US", prompt_text="Reference sentence."
    )
    assert prompt == "[EN]Reference sentence.\n"
    assert target == "New sentence."
    prompt, target = prepare_conditioned_text(
        "New sentence.", language="yue", prompt_text=None
    )
    assert prompt == ""
    assert target == "[口音:粤语]New sentence."


def test_schedule_rejects_invalid_budgets_and_templates() -> None:
    with pytest.raises(ValueError, match="at least one audio patch"):
        build_generation_schedule(
            text="[EN]Oracle fixture sentence.",
            tokenizer=_tokenizer(),
            max_audio_patches=6,
            template="tts_interleave",
        )
    with pytest.raises(ValueError, match="unsupported"):
        build_generation_schedule(
            text="[EN]Oracle fixture sentence.",
            tokenizer=_tokenizer(),
            max_audio_patches=8,
            template="unknown",  # type: ignore[arg-type]
        )
