"""Drama 3 prompt rows, as upstream fish_omni.generate.build_drama3_prompt.

A prompt is a system turn (with voice references), completed history turns, and the
current user turn, ending in an open assistant voice turn. Text rows are
[token, 0, ..., 0]; audio rows are [code 0 + semantic_start, code 0, ..., code N-1].
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from tokenizers import Tokenizer

from .config import FishDrama3Config

TTS_INSTRUCTION = "convert the provided text to speech"
REFERENCE_INSTRUCTION = (
    "convert the provided text to speech reference to the following:"
)
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
# Exact system instruction from the technical report (Figure 14). Keeping the
# newline before <|fim_prefix|> avoids EOS stalls observed on Mandarin infills
# with a single-line variant.
INFILL_INSTRUCTION = (
    "fill in the missing part of the speech; "
    f"the suffix follows {FIM_SUFFIX} and the prefix follows\n{FIM_PREFIX}"
)
MESSAGE_END = "<|im_end|>\n"
GENERATION_START = "<|im_start|>assistant\n<|voice|>"
# Only the prompt builder writes these; other special tokens in caller text are content.
STRUCTURAL_TOKENS = (
    "<|im_start|>",
    "<|im_end|>",
    "<|voice|>",
    "<|text|>",
    "<|interleave|>",
    "<|semantic:",
    FIM_PREFIX,
    FIM_SUFFIX,
    "<|fim_middle|>",
    "<|fim_pad|>",
)
SPEAKER_TAG_PATTERN = re.compile(r"<\|speaker:([0-9]+)\|>")
MAX_SPEAKER_ID = 15
# The Unicode White_Space property. str.strip() without arguments also strips
# U+001C..U+001F, which would change the empty check and the speaker-tag rule.
UNICODE_WHITE_SPACE = "".join(
    map(
        chr,
        (*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B))
        + (0x2028, 0x2029, 0x202F, 0x205F, 0x3000),
    )
)
MAX_PROMPT_ROWS = 8192


class PromptTooLongError(ValueError):
    """The prompt has more than MAX_PROMPT_ROWS rows."""


def validate_text(text: str, location: str) -> None:
    """Non-empty, no structural tokens, and only canonical speaker tags 0..15."""
    if not text.strip(UNICODE_WHITE_SPACE):
        raise ValueError(f"{location}: text must not be empty")
    for token in STRUCTURAL_TOKENS:
        if token in text:
            raise ValueError(f"{location}: {token} is written by the prompt builder")
    for marker in re.finditer(r"<\|speaker", text):
        tag = SPEAKER_TAG_PATTERN.match(text, marker.start())
        digits = tag.group(1) if tag else ""
        if not digits or str(int(digits)) != digits or int(digits) > MAX_SPEAKER_ID:
            raise ValueError(
                f"{location}: speaker tags must be <|speaker:0|> to "
                f"<|speaker:{MAX_SPEAKER_ID}|>; got {text!r}"
            )


def leads_with_speaker_tag(text: str) -> bool:
    return SPEAKER_TAG_PATTERN.match(text.lstrip(UNICODE_WHITE_SPACE)) is not None


def with_speaker_tag(text: str, speaker: int = 0) -> str:
    """Prefix <|speaker:speaker|> unless a tag already leads the text."""
    return text if leads_with_speaker_tag(text) else f"<|speaker:{speaker}|> {text}"


def with_duration_hint(text: str, frames: int) -> str:
    """Ask for audio of about frames frames (technical report, Table 10).

    The model ends close to the hint, so a hint shorter than the text needs drops
    its last words; one hint covers the whole text.
    """
    if frames < 1:
        raise ValueError(f"a duration hint needs at least one frame; got {frames}")
    return f"{text.rstrip(chr(10))} [audio tokens: {frames}]"


def _validate_prompt_setup(
    tokenizer: Tokenizer, config: FishDrama3Config
) -> tuple[int, int]:
    """Verify the tokenizer's semantic band and EOS id match config."""
    num_codebooks = config.audio_decoder.num_codebooks
    codebook_size = config.audio_decoder.vocab_size
    for token, expected in (
        ("<|semantic:0|>", config.semantic_start_token_id),
        (f"<|semantic:{codebook_size - 1}|>", config.semantic_end_token_id),
        ("<|im_end|>", config.eos_token_id),
    ):
        if tokenizer.token_to_id(token) != expected:
            raise ValueError(f"tokenizer/config mismatch for {token}")
    return num_codebooks, codebook_size


def _check_codes(
    codes: np.ndarray,
    location: str,
    num_codebooks: int,
    codebook_size: int,
) -> None:
    if (
        not np.issubdtype(codes.dtype, np.integer)
        or codes.ndim != 2
        or codes.shape[0] != num_codebooks
        or codes.shape[1] == 0
    ):
        raise ValueError(
            f"{location}: codes must be integer [{num_codebooks}, frames >= 1]; "
            f"got {codes.dtype} {codes.shape}"
        )
    if codes.min() < 0 or codes.max() >= codebook_size:
        raise ValueError(f"{location}: codes must be in [0, {codebook_size})")


def _encode_segments(
    segments: Sequence[str | np.ndarray],
    tokenizer: Tokenizer,
    config: FishDrama3Config,
) -> np.ndarray:
    num_codebooks = config.audio_decoder.num_codebooks
    rows = []
    for segment in segments:
        if isinstance(segment, str):
            ids = tokenizer.encode(segment).ids
            text_rows = np.zeros((len(ids), 1 + num_codebooks), dtype=np.int32)
            text_rows[:, 0] = ids
            rows.append(text_rows)
        else:
            codes = segment.astype(np.int32)
            main = codes[:1] + config.semantic_start_token_id
            rows.append(np.concatenate([main, codes]).T)
    prompt = np.concatenate(rows)
    if len(prompt) > MAX_PROMPT_ROWS:
        raise PromptTooLongError(
            f"prompt has {len(prompt)} rows; at most {MAX_PROMPT_ROWS}"
        )
    return prompt


def build_prompt(
    text: str,
    tokenizer: Tokenizer,
    config: FishDrama3Config,
    reference_texts: Sequence[str] = (),
    reference_codes: Sequence[np.ndarray] = (),
    history: Sequence[tuple[str, np.ndarray]] = (),
) -> np.ndarray:
    """int32 rows [rows, 1 + num_codebooks].

    Every literal is tokenized on its own: tokenizing the joined string changes ids
    at the seams. history holds earlier (text, codes) chunks, written as completed
    user and assistant voice turns before the current one.
    """
    num_codebooks, codebook_size = _validate_prompt_setup(tokenizer, config)
    if len(reference_texts) != len(reference_codes):
        raise ValueError("every reference needs both a transcript and codes")
    validate_text(text, "text")

    segments: list[str | np.ndarray] = [
        "<|im_start|>system\n",
        REFERENCE_INSTRUCTION if reference_codes else TTS_INSTRUCTION,
    ]
    for index, (transcript, codes) in enumerate(
        zip(reference_texts, reference_codes), 1
    ):
        location = f"reference {index}"
        validate_text(transcript, location)
        _check_codes(codes, location, num_codebooks, codebook_size)
        # The tag must follow "Text: " directly; leading white space moves behind it.
        tagged = with_speaker_tag(transcript)
        body = tagged.lstrip(UNICODE_WHITE_SPACE)
        tag = SPEAKER_TAG_PATTERN.match(body)
        leading = tagged[: len(tagged) - len(body)]
        transcript = tag.group(0) + leading + body[tag.end() :]
        segments += [f"\n\n[Reference {index}]\nText: {transcript}\nSpeech: ", codes]
    segments.append(MESSAGE_END)
    for index, (turn_text, codes) in enumerate(history, 1):
        location = f"history {index}"
        validate_text(turn_text, location)
        _check_codes(codes, location, num_codebooks, codebook_size)
        segments += [
            "<|im_start|>user\n",
            with_speaker_tag(turn_text),
            MESSAGE_END,
            GENERATION_START,
            codes,
            MESSAGE_END,
        ]
    segments += [
        "<|im_start|>user\n",
        with_speaker_tag(text),
        MESSAGE_END,
        GENERATION_START,
    ]
    return _encode_segments(segments, tokenizer, config)


def build_infill_prompt(
    text: str,
    prefix_codes: np.ndarray,
    suffix_codes: np.ndarray,
    tokenizer: Tokenizer,
    config: FishDrama3Config,
) -> np.ndarray:
    """int32 rows [rows, 1 + num_codebooks] for audio infilling (technical report,
    Section 5.5 and Figure 14).

    text is the full target transcript after the edit; prefix_codes and
    suffix_codes are the kept audio frames before and after the gap.
    """
    num_codebooks, codebook_size = _validate_prompt_setup(tokenizer, config)
    validate_text(text, "text")
    _check_codes(prefix_codes, "prefix_codes", num_codebooks, codebook_size)
    _check_codes(suffix_codes, "suffix_codes", num_codebooks, codebook_size)
    segments: list[str | np.ndarray] = [
        "<|im_start|>system\n",
        INFILL_INSTRUCTION,
        MESSAGE_END,
        "<|im_start|>user\n",
        with_speaker_tag(text),
        "\n",
        FIM_SUFFIX,
        suffix_codes,
        FIM_PREFIX,
        prefix_codes,
        MESSAGE_END,
        GENERATION_START,
    ]
    return _encode_segments(segments, tokenizer, config)
