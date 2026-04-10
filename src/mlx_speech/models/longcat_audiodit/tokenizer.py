"""Local tokenizer wrapper for LongCat AudioDiT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer


class _TokenizerBackend:
    def __init__(self, tokenizer: Tokenizer, *, pad_token_id: int) -> None:
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id

    def __call__(self, texts: list[str], **kwargs) -> dict[str, list[list[int]]]:
        padding = kwargs.get("padding", "longest")
        return_tensors = kwargs.get("return_tensors")
        if padding != "longest":
            raise ValueError(f"Unsupported padding mode: {padding}")
        if return_tensors is not None:
            raise ValueError("LongCatTokenizer only supports return_tensors=None.")

        encodings = self.tokenizer.encode_batch(texts)
        input_ids = [encoding.ids for encoding in encodings]
        max_length = max((len(ids) for ids in input_ids), default=0)
        padded_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for ids in input_ids:
            pad = max_length - len(ids)
            padded_ids.append(ids + ([self.pad_token_id] * pad))
            attention_mask.append(([1] * len(ids)) + ([0] * pad))
        return {"input_ids": padded_ids, "attention_mask": attention_mask}


class AutoTokenizer:
    @classmethod
    def from_pretrained(
        cls,
        path: str | Path,
        *,
        local_files_only: bool = True,
        use_fast: bool = False,
    ) -> _TokenizerBackend:
        if not local_files_only:
            raise ValueError("LongCat tokenizer loader is local-files-only.")
        del use_fast

        resolved = Path(path)
        config_dir = resolved if resolved.is_dir() else resolved.parent
        tokenizer_path = resolved if resolved.is_file() else resolved / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        pad_token_id = 0
        config_path = config_dir / "config.json"
        if config_path.exists():
            with config_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            pad_token_id = int(payload.get("pad_token_id", pad_token_id))

        return _TokenizerBackend(
            Tokenizer.from_file(str(tokenizer_path)), pad_token_id=pad_token_id
        )


class LongCatTokenizer:
    def __init__(self, backend: _TokenizerBackend) -> None:
        self.backend = backend

    @classmethod
    def from_path(cls, path: str | Path) -> "LongCatTokenizer":
        backend = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, use_fast=False
        )
        return cls(backend)

    def encode_text(self, texts: list[str]) -> dict[str, list[list[int]]]:
        return self.backend(texts, padding="longest", return_tensors=None)
