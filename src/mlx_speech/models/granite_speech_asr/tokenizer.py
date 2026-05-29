"""Tokenizer and prompt rendering for Granite Speech ASR."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TRANSCRIPTION_PROMPT = "can you transcribe the speech into a written format?"
AUDIO_TOKEN = "<|audio|>"


class GraniteSpeechTokenizer:
    """Thin wrapper around Granite's GPT-2 BPE tokenizer assets."""

    def __init__(self, tokenizer_path: str | Path, added_tokens_path: str | Path, chat_template_path: str | Path):
        try:
            from tokenizers import Tokenizer
        except ImportError as e:
            raise ImportError(
                "The `tokenizers` package is required. Install with: pip install tokenizers"
            ) from e

        tokenizer_path = Path(tokenizer_path)
        added_tokens_path = Path(added_tokens_path)
        chat_template_path = Path(chat_template_path)
        for path in (tokenizer_path, added_tokens_path, chat_template_path):
            if not path.exists():
                raise FileNotFoundError(f"Granite tokenizer asset not found: {path}")

        self._tok = Tokenizer.from_file(str(tokenizer_path))
        with added_tokens_path.open(encoding="utf-8") as f:
            self.added_tokens: dict[str, int] = {str(k): int(v) for k, v in json.load(f).items()}
        self.chat_template = chat_template_path.read_text(encoding="utf-8")

        self.audio_token = AUDIO_TOKEN
        self.audio_token_id = self.added_tokens.get(AUDIO_TOKEN)
        tokenizer_audio_id = self._tok.token_to_id(AUDIO_TOKEN)
        if self.audio_token_id is None:
            raise ValueError(f"{AUDIO_TOKEN!r} missing from {added_tokens_path}")
        if tokenizer_audio_id != self.audio_token_id:
            raise ValueError(
                f"{AUDIO_TOKEN!r} id mismatch: tokenizer={tokenizer_audio_id!r} "
                f"added_tokens={self.audio_token_id!r}"
            )

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "GraniteSpeechTokenizer":
        model_dir = Path(model_dir)
        return cls(
            model_dir / "tokenizer.json",
            model_dir / "added_tokens.json",
            model_dir / "chat_template.jinja",
        )

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoding = self._tok.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def render_prompt(
        self,
        num_audio_tokens: int,
        user_prompt: str | None = None,
    ) -> str:
        """Render the Granite speech chat prompt used before generation."""
        if num_audio_tokens < 0:
            raise ValueError("num_audio_tokens must be non-negative")
        prompt = user_prompt if user_prompt is not None else DEFAULT_TRANSCRIPTION_PROMPT
        content = f"{self.audio_token * num_audio_tokens}{prompt}"
        return f"USER: {content}\n ASSISTANT:"

    def build_prompt_ids(
        self,
        num_audio_tokens: int,
        user_prompt: str | None = None,
    ) -> list[int]:
        return self.encode(self.render_prompt(num_audio_tokens, user_prompt))
