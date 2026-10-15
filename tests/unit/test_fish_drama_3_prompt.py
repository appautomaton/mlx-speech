"""Drama 3 prompt rows and chunking, ported from upstream's test cases."""

import re
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from mlx_speech.models.fish_drama_3 import FishDrama3Config
from mlx_speech.models.fish_drama_3.chunking import (
    ChunkHistory,
    split_by_length,
    split_text,
    utf8_len,
)
from mlx_speech.models.fish_drama_3.prompt import build_prompt, with_duration_hint
from tests.unit.test_fish_drama_3_config import tiny_payload

CONFIG = FishDrama3Config.from_dict(tiny_payload())  # band 200..263, 3 codebooks


class RecordingTokenizer:
    """Records every literal it encodes; each one becomes two tokens."""

    def __init__(self) -> None:
        self.parts: list[str] = []

    def token_to_id(self, token: str) -> int | None:
        return {"<|semantic:0|>": 200, "<|semantic:63|>": 263, "<|im_end|>": 150}.get(
            token
        )

    def encode(self, text: str) -> SimpleNamespace:
        self.parts.append(text)
        return SimpleNamespace(ids=[3, 4])


def test_each_literal_is_tokenized_on_its_own():
    tokenizer = RecordingTokenizer()

    rows = build_prompt("hello", tokenizer, CONFIG)

    assert tokenizer.parts == [
        "<|im_start|>system\n",
        "convert the provided text to speech",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        "<|speaker:0|> hello",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    assert rows.shape == (14, 4) and not rows[:, 1:].any()


def test_reference_codes_become_audio_rows_and_whitespace_moves_behind_the_tag():
    tokenizer = RecordingTokenizer()
    codes = np.array([[1, 2], [3, 4], [5, 6]])

    rows = build_prompt(
        "<|speaker:0|>hello", tokenizer, CONFIG, ["  <|speaker:0|> reference"], [codes]
    )

    assert (
        "convert the provided text to speech reference to the following:"
        in tokenizer.parts
    )
    assert (
        "\n\n[Reference 1]\nText: <|speaker:0|>   reference\nSpeech: "
        in tokenizer.parts
    )
    audio = rows[(rows[:, 0] >= 200) & (rows[:, 0] <= 263)]
    np.testing.assert_array_equal(audio[:, 1:].T, codes)
    np.testing.assert_array_equal(audio[:, 0], codes[0] + 200)


def test_history_is_written_as_completed_turns_and_speakers_are_kept():
    tokenizer = RecordingTokenizer()
    codes = np.array([[1, 2], [3, 4], [5, 6]])
    text = "<|speaker:15|> hi\n<|speaker:0|> yes"

    plain = build_prompt(text, RecordingTokenizer(), CONFIG)
    rows = build_prompt(
        text, tokenizer, CONFIG, history=[("<|speaker:3|> before", codes)]
    )

    assert tokenizer.parts[-9:] == [
        "<|im_start|>user\n",
        "<|speaker:3|> before",
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
        "<|im_end|>\n",
        "<|im_start|>user\n",
        text,
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|voice|>",
    ]
    assert len(rows) - len(plain) == 5 * 2 + 2  # five literals, two audio frames


@pytest.mark.parametrize(
    "text",
    [
        " ",
        "hello <|voice|>",
        "<|speaker:16|> hi",
        "<|speaker:00|> hi",
        "<|speaker:x|> hi",
    ],
)
def test_rejects_invalid_text(text):
    with pytest.raises(ValueError):
        build_prompt(text, RecordingTokenizer(), CONFIG)


@pytest.mark.parametrize(
    "codes",
    [np.zeros((2, 3), dtype=np.int64), np.full((3, 2), 64), np.zeros((3, 2))],
)
def test_rejects_invalid_reference_codes(codes):
    with pytest.raises(ValueError):
        build_prompt("hi", RecordingTokenizer(), CONFIG, ["ref"], [codes])


def test_rejects_a_tokenizer_that_disagrees_with_the_config():
    with pytest.raises(ValueError, match="tokenizer/config"):
        build_prompt("hello", RecordingTokenizer(), replace(CONFIG, eos_token_id=151))


def test_long_text_splits_at_sentence_ends_and_reassembles():
    text = " ".join(f"Sentence number {i} is here." for i in range(40))
    chunks = split_by_length(text, 100)
    assert len(chunks) > 1 and " ".join(chunks) == text
    assert all(utf8_len(c) <= 100 and c.endswith(".") for c in chunks)

    text = "今天天气很好，我们去公园散步。" * 20
    chunks = split_by_length(text, 100)
    assert "".join(chunks) == text and all(c.endswith("。") for c in chunks)


def test_decimals_and_protected_spans_stay_whole():
    whisper = "<whisper>" + "secret words. " * 10 + "</whisper>"
    chunks = split_by_length(
        "Pi is 3.14159 roughly. " * 10 + whisper + " [laughing] done.", 60
    )
    assert whisper in chunks
    assert not any(c.endswith("3.") for c in chunks)
    assert any("[laughing]" in c for c in chunks)


def test_a_chunk_holds_at_most_three_speakers():
    text = "<|speaker:0|>A1<|speaker:1|>B1<|speaker:0|>A2<|speaker:2|>C1<|speaker:3|>D1"
    assert split_text(text) == [
        "<|speaker:0|> A1\n<|speaker:1|> B1\n<|speaker:0|> A2\n<|speaker:2|> C1",
        "<|speaker:3|> D1",
    ]

    script = "".join(
        f"<|speaker:{i % 5}|>Line {i} of the scene goes here. " for i in range(30)
    )
    for chunk in split_text(script):
        build_prompt(chunk, RecordingTokenizer(), CONFIG)
        assert len(set(re.findall(r"speaker:(\d+)", chunk))) <= 3


def test_history_pins_the_first_chunk_and_keeps_the_latest():
    history = ChunkHistory(recent_turns=1)
    assert list(history.candidates()) == [[]]
    for index in range(3):
        history.push(f"chunk {index}", np.full((3, 1), index))

    texts = [[text for text, _ in turns] for turns in history.candidates()]

    assert texts == [["chunk 0", "chunk 2"], ["chunk 0"], []]


def test_duration_hint_closes_the_text_and_survives_validation():
    text = with_duration_hint("<|speaker:0|> 找小三！", 30)

    assert text == "<|speaker:0|> 找小三！ [audio tokens: 30]"
    build_prompt(text, RecordingTokenizer(), CONFIG)
    with pytest.raises(ValueError, match="at least one frame"):
        with_duration_hint("hi", 0)
