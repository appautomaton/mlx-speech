"""Text normalization helpers for LongCat AudioDiT."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'["“”‘’]', " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def approx_duration_from_text(text: str, max_duration: float = 30.0) -> float:
    en_dur_per_char = 0.082
    zh_dur_per_char = 0.21
    text = re.sub(r"\s+", "", text)
    num_zh = 0
    num_en = 0
    num_other = 0
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            num_zh += 1
        elif char.isalpha():
            num_en += 1
        else:
            num_other += 1
    if num_zh > num_en:
        num_zh += num_other
    else:
        num_en += num_other
    return min(max_duration, (num_zh * zh_dur_per_char) + (num_en * en_dur_per_char))
