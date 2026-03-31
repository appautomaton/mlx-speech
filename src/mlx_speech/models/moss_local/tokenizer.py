"""Tokenizer helpers for MossTTSLocal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

DEFAULT_MOSS_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n"
    "{% if message['content'] is string %}"
    "{{ message['content'] }}"
    "{% else %}"
    "{% for content in message['content'] %}"
    "{% if content.get('type') == 'text' %}{{ content['text'] }}{% endif %}"
    "{% endfor %}"
    "{% endif %}"
    "<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


@dataclass(frozen=True)
class MossTTSLocalTokenizer:
    """Small wrapper around the local Hugging Face tokenizer assets."""

    tokenizer: Tokenizer
    model_dir: Path
    chat_template: str = DEFAULT_MOSS_CHAT_TEMPLATE

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "MossTTSLocalTokenizer":
        resolved_dir = Path(model_dir)
        tokenizer_path = resolved_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        tokenizer_config_path = resolved_dir / "tokenizer_config.json"
        chat_template = DEFAULT_MOSS_CHAT_TEMPLATE
        if tokenizer_config_path.exists():
            with tokenizer_config_path.open(encoding="utf-8") as f:
                payload = json.load(f)
            chat_template = str(payload.get("chat_template") or DEFAULT_MOSS_CHAT_TEMPLATE)
        return cls(
            tokenizer=Tokenizer.from_file(str(tokenizer_path)),
            model_dir=resolved_dir,
            chat_template=chat_template,
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Tokenizer does not define token: {token}")
        return int(token_id)

    @staticmethod
    def _render_content(content: str | list[dict[str, Any]]) -> str:
        if isinstance(content, str):
            return content

        rendered_parts: list[str] = []
        for item in content:
            if item.get("type") == "text":
                rendered_parts.append(str(item.get("text", "")))
        return "".join(rendered_parts)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        add_generation_prompt: bool,
        tokenize: bool = False,
    ) -> str | list[int]:
        rendered = []
        for message in messages:
            role = str(message["role"])
            content = self._render_content(message["content"])
            rendered.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            rendered.append("<|im_start|>assistant\n")

        text = "".join(rendered)
        if tokenize:
            return self.encode(text)
        return text
