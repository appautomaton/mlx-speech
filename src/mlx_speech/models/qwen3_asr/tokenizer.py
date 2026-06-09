"""Tokenizer asset loading for Qwen3-ASR."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Qwen3ASRConfig


@dataclass(frozen=True)
class Qwen3ASRTokenizer:
    """Thin wrapper around Qwen's byte-level BPE tokenizer assets."""

    tokenizer: Any
    audio_token: str
    audio_bos_token: str
    audio_eos_token: str
    audio_token_id: int
    audio_bos_token_id: int
    audio_eos_token_id: int
    eos_token: str
    eos_token_id: int
    pad_token: str
    pad_token_id: int
    model_max_length: int

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Qwen3ASRTokenizer":
        model_dir = Path(model_dir)
        config = Qwen3ASRConfig.from_dir(model_dir)
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        vocab_path = model_dir / "vocab.json"
        merges_path = model_dir / "merges.txt"
        for path in (tokenizer_config_path, vocab_path, merges_path):
            if not path.exists():
                raise FileNotFoundError(f"Qwen3-ASR tokenizer asset not found: {path}")

        with tokenizer_config_path.open(encoding="utf-8") as f:
            tokenizer_config = json.load(f)

        tokenizer = _load_bpe_tokenizer(
            vocab_path=vocab_path,
            merges_path=merges_path,
            tokenizer_config=tokenizer_config,
        )

        audio_token = str(tokenizer_config["audio_token"])
        audio_bos_token = str(tokenizer_config["audio_bos_token"])
        audio_eos_token = str(tokenizer_config["audio_eos_token"])
        eos_token = str(tokenizer_config["eos_token"])
        pad_token = str(tokenizer_config["pad_token"])

        audio_token_id = _require_token_id(tokenizer, audio_token)
        audio_bos_token_id = _require_token_id(tokenizer, audio_bos_token)
        audio_eos_token_id = _require_token_id(tokenizer, audio_eos_token)
        eos_token_id = _require_token_id(tokenizer, eos_token)
        pad_token_id = _require_token_id(tokenizer, pad_token)

        if audio_token_id != config.audio_token_id:
            raise ValueError(
                f"audio_token_id mismatch: tokenizer={audio_token_id}, "
                f"config={config.audio_token_id}"
            )
        if audio_bos_token_id != config.audio_start_token_id:
            raise ValueError(
                f"audio_bos_token_id mismatch: tokenizer={audio_bos_token_id}, "
                f"config={config.audio_start_token_id}"
            )
        if audio_eos_token_id != config.audio_end_token_id:
            raise ValueError(
                f"audio_eos_token_id mismatch: tokenizer={audio_eos_token_id}, "
                f"config={config.audio_end_token_id}"
            )

        return cls(
            tokenizer=tokenizer,
            audio_token=audio_token,
            audio_bos_token=audio_bos_token,
            audio_eos_token=audio_eos_token,
            audio_token_id=audio_token_id,
            audio_bos_token_id=audio_bos_token_id,
            audio_eos_token_id=audio_eos_token_id,
            eos_token=eos_token,
            eos_token_id=eos_token_id,
            pad_token=pad_token,
            pad_token_id=pad_token_id,
            model_max_length=int(tokenizer_config.get("model_max_length", 0)),
        )

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
        ).ids

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def token_to_id(self, token: str) -> int | None:
        token_id = self.tokenizer.token_to_id(token)
        return int(token_id) if token_id is not None else None


def _load_bpe_tokenizer(
    *,
    vocab_path: Path,
    merges_path: Path,
    tokenizer_config: dict[str, Any],
) -> Any:
    try:
        from tokenizers import AddedToken, Tokenizer
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import ByteLevel
    except ImportError as e:
        raise ImportError(
            "The `tokenizers` package is required. Install with: pip install tokenizers"
        ) from e

    unk_token = tokenizer_config.get("unk_token")
    model = BPE.from_file(
        str(vocab_path),
        str(merges_path),
        unk_token=str(unk_token) if unk_token is not None else None,
    )
    tokenizer = Tokenizer(model)
    tokenizer.pre_tokenizer = ByteLevel(
        add_prefix_space=bool(tokenizer_config.get("add_prefix_space", False))
    )
    tokenizer.decoder = ByteLevelDecoder()

    added_tokens = []
    decoder_config = tokenizer_config.get("added_tokens_decoder", {})
    for item in decoder_config.values():
        added_tokens.append(
            AddedToken(
                str(item["content"]),
                single_word=bool(item.get("single_word", False)),
                lstrip=bool(item.get("lstrip", False)),
                rstrip=bool(item.get("rstrip", False)),
                normalized=bool(item.get("normalized", False)),
                special=bool(item.get("special", False)),
            )
        )
    tokenizer.add_special_tokens(added_tokens)
    return tokenizer


def _require_token_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Token {token!r} not found in Qwen3-ASR tokenizer")
    return int(token_id)
