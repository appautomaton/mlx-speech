from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from mlx_speech.models.confucius4_r2t2.streaming import R2T2StreamSession


_SPLIT = "¤"


def _tokenize(text: str) -> list[int]:
    return [ord(char) for char in text]


def _detokenize(ids: Sequence[int]) -> str:
    return "".join(chr(int(token)) for token in ids)


def _tokenize_split(text: str) -> list[int]:
    ids: list[int] = []
    for char in text:
        if char == _SPLIT:
            ids.extend((0xE000, 0xE001))
        else:
            ids.append(ord(char))
    return ids


def _detokenize_split(ids: Sequence[int]) -> str:
    tokens = [int(token) for token in ids]
    chars: list[str] = []
    index = 0
    while index < len(tokens):
        if tokens[index] == 0xE000:
            if index + 1 < len(tokens) and tokens[index + 1] == 0xE001:
                chars.append(_SPLIT)
                index += 2
            else:
                chars.append("\ufffd")
                index += 1
        else:
            chars.append(chr(tokens[index]))
            index += 1
    return "".join(chars)


class _Script:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
        self.calls.append((prefix, int(audio.shape[0]), int(max_new_tokens)))
        if len(self.calls) > len(self.outputs):
            raise AssertionError("llm_generate called more times than scripted")
        return self.outputs[len(self.calls) - 1]


def _session(script: _Script, **kwargs) -> R2T2StreamSession:
    tokenize = kwargs.pop("tokenize", _tokenize)
    detokenize = kwargs.pop("detokenize", _detokenize)
    kwargs.setdefault("lookahead_ms", 0)
    kwargs.setdefault("chunk_ms", 80)
    return R2T2StreamSession(script, tokenize, detokenize, **kwargs)


def _pcm(samples: int, value: float = 0.0) -> np.ndarray:
    return np.full((samples,), value, dtype=np.float32)


def test_short_feed_waits_for_first_window_then_chunk():
    script = _Script(["hello", "hello!"])
    session = R2T2StreamSession(
        script,
        _tokenize,
        _detokenize,
        language="English",
        chunk_ms=160,
        lookahead_ms=160,
        unfixed_token_num=0,
    )
    short = session.feed(_pcm(1000))
    assert short.delta == ""
    assert script.calls == []

    first = session.feed(_pcm(5120 - 1000))
    assert len(script.calls) == 1
    assert script.calls[0][1] == 5120
    assert first.delta == "hello"

    session.feed(_pcm(2559))
    assert len(script.calls) == 1
    session.feed(_pcm(1))
    assert len(script.calls) == 2
    assert script.calls[1][1] == 5120 + 2560


def test_unfixed_token_is_missing_from_prefix_and_fixed_text():
    script = _Script(["abcd", "abcd"])
    session = _session(
        script, language="English", unfixed_token_num=1, unfixed_chunk_num=0
    )
    update = session.feed(_pcm(1280))
    assert update.hypothesis == "abcd"
    assert update.committed == "abc"
    assert update.delta == "abc"
    session.feed(_pcm(1280))
    assert script.calls[1][0] == "abc"


def test_replacement_char_drops_one_extra_prefix_token():
    script = _Script([f"ab{_SPLIT}", "tail"])
    session = _session(
        script,
        language="English",
        unfixed_token_num=1,
        tokenize=_tokenize_split,
        detokenize=_detokenize_split,
    )
    session.feed(_pcm(1280))
    session.feed(_pcm(1280))
    assert script.calls[1][0] == "ab"
    assert "\ufffd" not in script.calls[1][0]


def test_empty_asr_text_body_keeps_every_fixed_token():
    seen: list[list[int]] = []

    def detokenize(ids: Sequence[int]) -> str:
        seen.append([int(token) for token in ids])
        return _detokenize(ids)

    raw = "language Chinese<asr_text>"
    script = _Script([raw])
    session = _session(
        script, language=None, unfixed_token_num=1, detokenize=detokenize
    )
    update = session.feed(_pcm(1280))
    assert update.hypothesis == ""
    assert update.committed == ""
    assert seen[-1] == _tokenize(raw)


def test_prefix_and_fixed_punctuation_sets_differ():
    comma = _Script(["hello,", "next"])
    comma_session = _session(
        comma, language="English", rollback_punctuation=True, unfixed_token_num=1
    )
    comma_update = comma_session.feed(_pcm(1280))
    assert comma_update.committed == "hello,"
    comma_session.feed(_pcm(1280))
    assert comma.calls[1][0] == "hello"

    period = _Script(["hello.", "next"])
    period_session = _session(
        period, language="English", rollback_punctuation=True, unfixed_token_num=1
    )
    period_session.feed(_pcm(1280))
    period_session.feed(_pcm(1280))
    assert period.calls[1][0] == "hello."


def test_missing_tag_clears_public_text_and_keeps_audio():
    script = _Script(["hello", "language English<asr_text>yo"])
    session = _session(script, language=None, unfixed_token_num=1)
    update = session.feed(_pcm(1280))
    assert update.hypothesis == ""
    assert update.delta == ""
    assert session.chunk_id == 0
    assert session.audio_samples == 1280

    session.feed(_pcm(1280))
    assert script.calls[1][0] == "hell"
    assert script.calls[1][1] == 2560
    assert session.chunk_id == 1


def test_append_only_lock_keeps_the_old_head():
    script = _Script(["hello", "HELLO!!", "hi"])
    session = _session(
        script, language="English", unfixed_token_num=0, unfixed_chunk_num=5
    )
    first = session.feed(_pcm(1280))
    assert first.committed == "hello"
    second = session.feed(_pcm(1280))
    assert second.hypothesis == "HELLO!!"
    assert second.delta == "!!"
    assert second.committed == "hello!!"
    third = session.feed(_pcm(1280))
    assert third.delta == ""
    assert third.committed == "hello!!"


def test_pipe_cut_and_punctuation_and_cjk_spaces():
    pipes = _session(_Script(["foo|bar"]), language="English", unfixed_token_num=0)
    piped = pipes.feed(_pcm(1280))
    assert piped.hypothesis == "foo"
    assert piped.committed == "foo"

    chinese_mark = _session(_Script(["你."]), language="Chinese", unfixed_token_num=0)
    assert chinese_mark.feed(_pcm(1280)).hypothesis == "你。"

    english_mark = _session(_Script(["a。"]), language="English", unfixed_token_num=0)
    assert english_mark.feed(_pcm(1280)).hypothesis == "a."

    kept = _session(_Script(["hello 世界"]), language="Chinese", unfixed_token_num=0)
    assert kept.feed(_pcm(1280)).hypothesis == "hello 世界"

    deleted = _session(_Script(["你 好"]), language="Chinese", unfixed_token_num=0)
    assert deleted.feed(_pcm(1280)).hypothesis == "你好"


def test_empty_finalize_commits_held_back_token_without_generate():
    script = _Script(["abcd"])
    session = _session(script, language="English", unfixed_token_num=1)
    update = session.feed(_pcm(1280))
    assert update.committed == "abc"
    assert update.hypothesis == "abcd"
    chunk_id = session.chunk_id
    result = session.finalize()
    assert script.calls == [("", 1280, 1)]
    assert result.text == "abcd"
    assert session.chunk_id == chunk_id


def test_finish_prefix_skips_per_chunk_rollback_rules():
    punct = _Script(["TAIL"])
    punct_session = _session(
        punct, language="English", unfixed_token_num=1, rollback_punctuation=True
    )
    punct_session._raw_decoded = "你好。"
    punct_session.feed(_pcm(10))
    punct_session.finalize()
    assert punct.calls[0][0] == "你好"

    pipe = _Script(["TAIL"])
    pipe_session = _session(pipe, language="English", unfixed_token_num=1)
    pipe_session._raw_decoded = "ab|cd"
    pipe_session.feed(_pcm(10))
    pipe_session.finalize()
    assert pipe.calls[0][0] == "ab"

    broken = _Script(["TAIL"])
    broken_session = _session(
        broken,
        language="English",
        unfixed_token_num=1,
        tokenize=_tokenize_split,
        detokenize=_detokenize_split,
    )
    broken_session._raw_decoded = f"ab{_SPLIT}"
    broken_session.feed(_pcm(10))
    broken_session.finalize()
    assert broken.calls[0][0] == "ab\ufffd"


def test_finish_does_not_use_running_budget_or_cjk_space_deletion():
    script = _Script(["hello", "你 好"])
    session = _session(
        script,
        language="Chinese",
        unfixed_token_num=0,
        chunk_ms=80,
        lookahead_ms=160,
    )
    session.feed(_pcm(1280 + 2560))
    assert script.calls[0][2] == 3
    session.feed(_pcm(10))
    session.finalize()
    assert script.calls[1][2] == 3
    assert script.calls[1][0].endswith("hello")
    assert "你 好" in session._raw_decoded


def test_budget_resets_on_growth_and_keeps_half_steps_on_latin_stall():
    script = _Script(["hello", "", "", ""])
    session = _session(
        script, language="English", unfixed_token_num=0, chunk_ms=80, lookahead_ms=80
    )
    session.feed(_pcm(1280 * 5))
    assert [call[2] for call in script.calls] == [2, 1, 1, 2]


def test_cjk_stall_resets_then_doubles_without_adding_half():
    script = _Script(["你", "", ""])
    session = _session(
        script, language="Chinese", unfixed_token_num=0, chunk_ms=80, lookahead_ms=80
    )
    session.feed(_pcm(1280 * 4))
    assert [call[2] for call in script.calls] == [2, 2, 2]


def test_language_is_fixed_for_the_session():
    script = _Script(["你 好", "你 好"])
    session = _session(script, language="Chinese", unfixed_token_num=0)
    assert session.language == "Chinese"
    assert session.feed(_pcm(1280)).hypothesis == "你好"
    with pytest.raises(TypeError):
        session.feed(_pcm(1280), language="English")  # type: ignore[call-arg]
    assert session.feed(_pcm(1280)).hypothesis == "你好你好"
    assert session.language == "Chinese"


def test_chunk_ms_bounds_and_int16_scale():
    with pytest.raises(ValueError, match="80, 2000"):
        _session(_Script([]), chunk_ms=79)
    with pytest.raises(ValueError, match="80, 2000"):
        _session(_Script([]), chunk_ms=2001)

    script = _Script(["ok"])
    session = _session(script, language="English", unfixed_token_num=0)
    session.feed(np.full((1280,), 16384, dtype=np.int16))
    assert script.calls[0][1] == 1280
    assert script.calls[0][0] == ""


def test_int16_audio_is_scaled_before_generate():
    seen: list[np.ndarray] = []

    def generate(prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
        seen.append(audio)
        return "ok"

    session = R2T2StreamSession(
        generate,
        _tokenize,
        _detokenize,
        language="English",
        chunk_ms=80,
        lookahead_ms=0,
        unfixed_token_num=0,
    )
    session.feed(np.full((1280,), 16384, dtype=np.int16))
    assert seen[0].dtype == np.float32
    assert seen[0][0] == np.float32(0.5)


def test_one_feed_updates_budget_on_every_consumed_chunk():
    script = _Script(["hello", "hello"])
    session = _session(
        script, language="English", unfixed_token_num=0, chunk_ms=160, lookahead_ms=160
    )
    session.feed(_pcm(5120 + 2560))
    assert [call[2] for call in script.calls] == [4, 2]
    assert [call[1] for call in script.calls] == [5120, 7680]
