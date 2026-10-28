"""Tokenizer contract for Audio8 streaming ASR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Audio8Config


@dataclass(frozen=True)
class Audio8Tokenizer:
    tokenizer: Any
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    streaming_pad_token_id: int
    streaming_word_token_id: int
    language_token_ids: dict[str, int]

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Audio8Tokenizer":
        from tokenizers import Tokenizer

        model_dir = Path(model_dir)
        path = model_dir / "tokenizer.json"
        if not path.exists():
            raise FileNotFoundError(f"Audio8 tokenizer not found: {path}")
        tokenizer = Tokenizer.from_file(str(path))
        config = Audio8Config.from_dir(model_dir)
        result = cls(
            tokenizer=tokenizer,
            bos_token_id=_require_token_id(tokenizer, "<|im_start|>"),
            eos_token_id=_require_token_id(tokenizer, "<|im_end|>"),
            pad_token_id=_require_token_id(tokenizer, "<|endoftext|>"),
            streaming_pad_token_id=_require_token_id(tokenizer, "[STREAMING_PAD]"),
            streaming_word_token_id=_require_token_id(tokenizer, "[STREAMING_WORD]"),
            language_token_ids={
                "zh": _require_token_id(tokenizer, "[LANGUAGE_ZH]"),
                "en": _require_token_id(tokenizer, "[LANGUAGE_EN]"),
            },
        )
        expected = (config.bos_token_id, config.eos_token_id, config.pad_token_id)
        actual = (result.bos_token_id, result.eos_token_id, result.pad_token_id)
        if actual != expected:
            raise ValueError(f"Audio8 tokenizer/config special IDs differ: {actual} vs {expected}")
        return result

    def language_token_id(self, language: str) -> int:
        normalized = language.strip().lower()
        if normalized not in self.language_token_ids:
            raise ValueError("Audio8 language must be exactly 'zh' or 'en'")
        return self.language_token_ids[normalized]

    def visible_token_ids(self, token_ids: list[int] | tuple[int, ...]) -> list[int]:
        visible = []
        controls = {
            self.streaming_pad_token_id,
            self.streaming_word_token_id,
            self.bos_token_id,
            self.pad_token_id,
        }
        for token_id in token_ids:
            if token_id == self.eos_token_id:
                break
            if token_id not in controls:
                visible.append(int(token_id))
        return visible

    def decode_visible(self, token_ids: list[int] | tuple[int, ...]) -> str:
        return self.tokenizer.decode(
            self.visible_token_ids(token_ids), skip_special_tokens=True
        )


def _require_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Audio8 tokenizer is missing {token!r}")
    return int(token_id)
