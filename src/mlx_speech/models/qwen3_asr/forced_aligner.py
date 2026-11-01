"""Text units, prompt, and timestamp decoding for Qwen3-ForcedAligner.

The aligner is a Qwen3-ASR checkpoint whose ``lm_head`` predicts a time class
instead of a token. One causal forward pass over

    <|audio_start|> audio... <|audio_end|> u1 <ts><ts> u2 <ts><ts> ... uN <ts><ts>

yields a class at every ``<timestamp>`` position. Class times
``timestamp_segment_time`` is milliseconds, and each unit's two timestamps are
its start and end. This module mirrors upstream's ``Qwen3ForceAlignProcessor``
for Chinese and English. Japanese and Korean need external segmenters and are
not supported.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import Qwen3ASRConfig


SUPPORTED_ALIGN_LANGUAGES = ("zh", "en")
TIMESTAMP_TOKEN = "<timestamp>"
# Upstream MAX_FORCE_ALIGN_INPUT_SECONDS.
MAX_ALIGN_SECONDS = 180.0


@dataclass(frozen=True)
class Qwen3ForcedAlignerSettings:
    """Aligner fields that ``Qwen3ASRConfig`` keeps in its ``extra`` maps."""

    timestamp_token_id: int
    timestamp_segment_time_ms: float
    classify_num: int

    @classmethod
    def from_config(cls, config: Qwen3ASRConfig) -> "Qwen3ForcedAlignerSettings":
        thinker = config.thinker_config
        if not thinker.is_forced_aligner:
            raise ValueError(
                "Not a Qwen3-ForcedAligner checkpoint: thinker model_type is "
                f"{thinker.model_type!r}."
            )
        missing = [
            key
            for key in ("timestamp_token_id", "timestamp_segment_time")
            if key not in config.extra
        ]
        if missing:
            raise ValueError(f"Qwen3-ForcedAligner config.json is missing {missing}.")
        return cls(
            timestamp_token_id=int(config.extra["timestamp_token_id"]),
            timestamp_segment_time_ms=float(config.extra["timestamp_segment_time"]),
            classify_num=thinker.lm_head_size,
        )


def validate_align_language(language: str) -> str:
    if language not in SUPPORTED_ALIGN_LANGUAGES:
        raise ValueError(
            f"Qwen3-ForcedAligner language must be exactly 'zh' or 'en'; got {language!r}."
        )
    return language


def split_text_units(text: str) -> list[str]:
    """Split a transcript into alignment units.

    Whitespace separates words; only letters, numbers, and apostrophes survive,
    so punctuation inside a word joins its halves (``3.75`` becomes ``375``).
    Each CJK ideograph is its own unit and other runs stay whole:
    ``核心PCE`` becomes ``核``, ``心``, ``PCE``. Upstream applies this rule to
    Chinese and English alike, so mixed text aligns under either language.
    """

    units: list[str] = []
    for segment in text.split():
        buffer: list[str] = []
        for ch in segment:
            if not _is_kept_char(ch):
                continue
            if _is_cjk_char(ch):
                if buffer:
                    units.append("".join(buffer))
                    buffer = []
                units.append(ch)
            else:
                buffer.append(ch)
        if buffer:
            units.append("".join(buffer))
    return units


def build_aligner_prompt(
    units: Sequence[str],
    *,
    audio_length: int,
    audio_bos_token: str = "<|audio_start|>",
    audio_token: str = "<|audio_pad|>",
    audio_eos_token: str = "<|audio_end|>",
) -> str:
    """Aligner prompt with the audio placeholder expanded to ``audio_length``."""

    if not units:
        raise ValueError("Qwen3-ForcedAligner needs at least one text unit.")
    if audio_length <= 0:
        raise ValueError("audio_length must be positive.")
    pair = TIMESTAMP_TOKEN * 2
    return (
        audio_bos_token
        + audio_token * int(audio_length)
        + audio_eos_token
        + pair.join(units)
        + pair
    )


def fix_timestamps(values: Sequence[float] | np.ndarray) -> list[int]:
    """Make predicted timestamps monotonic, as upstream ``fix_timestamp`` does.

    The longest non-decreasing subsequence is kept, with upstream's tie-breaks:
    the earliest predecessor and the earliest end. Each run of other values is
    replaced: runs of one or two snap to the nearer kept neighbour, and longer
    runs are interpolated linearly between their kept neighbours. Results are
    truncated to whole milliseconds.
    """

    data = [float(value) for value in np.asarray(values, dtype=np.float64).reshape(-1)]
    n = len(data)
    if n == 0:
        return []

    # O(n^2) like upstream, but vectorized over predecessors. Invalid
    # predecessors score 0, so argmax picks the earliest best one.
    array = np.asarray(data, dtype=np.float64)
    lengths = np.ones(n, dtype=np.int64)
    parents = np.full(n, -1, dtype=np.int64)
    for i in range(1, n):
        candidates = np.where(array[:i] <= array[i], lengths[:i], 0)
        best = int(np.argmax(candidates))
        if candidates[best] > 0:
            lengths[i] = candidates[best] + 1
            parents[i] = best

    is_kept = [False] * n
    index = int(np.argmax(lengths))
    while index != -1:
        is_kept[index] = True
        index = int(parents[index])

    result = list(data)
    i = 0
    while i < n:
        if is_kept[i]:
            i += 1
            continue
        j = i
        while j < n and not is_kept[j]:
            j += 1
        # The run [i, j) is maximal, so i - 1 and j are kept when they exist.
        left = result[i - 1] if i > 0 else None
        right = result[j] if j < n else None
        count = j - i
        for k in range(i, j):
            if left is None:
                result[k] = right
            elif right is None:
                result[k] = left
            elif count <= 2:
                result[k] = left if (k - (i - 1)) <= (j - k) else right
            else:
                result[k] = left + (right - left) / (count + 1) * (k - i + 1)
        i = j
    return [int(value) for value in result]


def decode_timestamps(
    units: Sequence[str],
    timestamp_classes: Sequence[int] | np.ndarray,
    *,
    segment_time_ms: float,
) -> list[tuple[str, float, float]]:
    """Turn per-``<timestamp>`` classes into ``(unit, start_s, end_s)`` spans."""

    classes = np.asarray(timestamp_classes, dtype=np.int64).reshape(-1)
    if classes.size != 2 * len(units):
        raise ValueError(
            f"Expected {2 * len(units)} timestamps for {len(units)} units, got {classes.size}."
        )
    fixed_ms = fix_timestamps(classes.astype(np.float64) * float(segment_time_ms))
    return [
        (unit, round(fixed_ms[2 * i] / 1000.0, 3), round(fixed_ms[2 * i + 1] / 1000.0, 3))
        for i, unit in enumerate(units)
    ]


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    category = unicodedata.category(ch)
    return category.startswith("L") or category.startswith("N")


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
        or 0x3400 <= code <= 0x4DBF  # Extension A
        or 0x20000 <= code <= 0x2A6DF  # Extension B
        or 0x2A700 <= code <= 0x2B73F  # Extension C
        or 0x2B740 <= code <= 0x2B81F  # Extension D
        or 0x2B820 <= code <= 0x2CEAF  # Extension E
        or 0xF900 <= code <= 0xFAFF  # Compatibility Ideographs
    )
