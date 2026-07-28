"""Vocabulary decoding helpers for Nemotron's multilingual BPE tokens."""

from __future__ import annotations

import re

_LANGUAGE_TAG = re.compile(r"^<([a-z]{2,3}-[A-Za-z]{2,4})>$")
_OTHER_SPECIAL = frozenset({"<unk>", "<pad>", "<s>", "</s>", "<blank>"})


class NemotronTokenizer:
    def __init__(self, vocabulary: tuple[str, ...]) -> None:
        if not vocabulary:
            raise ValueError("Nemotron vocabulary must not be empty")
        self.vocabulary = vocabulary

    def piece(self, token_id: int) -> str | None:
        if 0 <= token_id < len(self.vocabulary):
            return self.vocabulary[token_id]
        return None

    def is_special(self, token_id: int) -> bool:
        piece = self.piece(token_id)
        return piece in _OTHER_SPECIAL or bool(piece and _LANGUAGE_TAG.fullmatch(piece))

    def detected_language(self, tokens: tuple[int, ...] | list[int]) -> str | None:
        for token in tokens:
            piece = self.piece(token)
            match = _LANGUAGE_TAG.fullmatch(piece or "")
            if match is not None:
                return match.group(1)
        return None

    def decode(
        self,
        tokens: tuple[int, ...] | list[int],
        *,
        strip_language_tags: bool = True,
    ) -> str:
        pieces = []
        for token in tokens:
            piece = self.piece(token)
            if piece is None or piece in _OTHER_SPECIAL:
                continue
            if strip_language_tags and _LANGUAGE_TAG.fullmatch(piece):
                continue
            pieces.append(piece.replace("▁", " "))
        return "".join(pieces).strip()


__all__ = ["NemotronTokenizer"]
