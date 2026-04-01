"""Step-Audio-EditX fast tokenizer and prompt helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer

try:
    from ..step_audio_tokenizer import format_audio_token_string
except ImportError:  # pragma: no cover - direct file loading in tests
    from mlx_speech.models.step_audio_tokenizer import format_audio_token_string


DEFAULT_STEP_AUDIO_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{{ '<s>' }}{% endif %}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}{% set role = 'human' %}"
    "{% else %}{% set role = message['role'] %}{% endif %}"
    "{{ '<|BOT|> ' + role + '\\n' }}"
    "{{ message['content'] }}"
    "{% if not loop.last or message['role'] != 'assistant' %}{{ '<|EOT|>' }}{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|BOT|> assistant\\n' }}{% endif %}"
)

AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL = """Generate audio with the following timbre, prosody and speaking style

[speaker_start]
speaker name: {speaker}
speaker prompt text: 
{prompt_text}
speaker audio tokens: 
{prompt_wav_tokens}
[speaker_end]
"""

AUDIO_EDIT_SYSTEM_PROMPT = """As a highly skilled audio editing and tuning specialist, you excel in interpreting user instructions and applying precise adjustments to meet their needs. Your expertise spans a wide range of enhancement capabilities, including but not limited to:
# Emotional Enhancement
# Speaking Style Transfer
# Non-linguistic Adjustments
# Audio Tuning & Editing
Note: You will receive instructions in natural language and are expected to accurately interpret and execute the most suitable audio edits and enhancements.
"""


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer config not found: {path}")
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid tokenizer config: {path}")
    return payload


def _normalize_paths(model_path: str | Path) -> tuple[Path, Path]:
    path = Path(model_path)
    if path.is_dir():
        tokenizer_path = path / "tokenizer.json"
        tokenizer_config = path / "tokenizer_config.json"
    else:
        tokenizer_path = path
        tokenizer_config = path.with_name("tokenizer_config.json")
    return tokenizer_path, tokenizer_config

def build_clone_messages(
    *,
    speaker: str,
    prompt_text: str,
    prompt_wav_tokens: str,
    target_text: str,
) -> list[dict[str, str]]:
    system_prompt = AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL.format(
        speaker=speaker,
        prompt_text=prompt_text,
        prompt_wav_tokens=prompt_wav_tokens,
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": target_text},
    ]


def build_edit_messages(
    *,
    instruct_prefix: str,
    audio_token_str: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": AUDIO_EDIT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruct_prefix}\n{audio_token_str}\n"},
    ]


@dataclass
class StepAudioEditXTokenizer:
    """Fast-tokenizer wrapper for Step-Audio-EditX."""

    tokenizer: Tokenizer
    model_dir: Path
    tokenizer_path: Path
    tokenizer_config_path: Path
    chat_template: str = DEFAULT_STEP_AUDIO_CHAT_TEMPLATE
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    pad_token: str = "<unk>"
    unk_token: str = "<unk>"
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    @classmethod
    def from_path(cls, model_dir: str | Path) -> StepAudioEditXTokenizer:
        resolved_dir = Path(model_dir)
        tokenizer_path, tokenizer_config_path = _normalize_paths(resolved_dir)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found: {tokenizer_path}. "
                "Generate the Step-Audio fast tokenizer before loading this family."
            )

        payload = _load_json(tokenizer_config_path)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        chat_template = str(payload.get("chat_template") or DEFAULT_STEP_AUDIO_CHAT_TEMPLATE)
        bos_token = str(payload.get("bos_token", "<s>"))
        eos_token = str(payload.get("eos_token", "</s>"))
        pad_token = str(payload.get("pad_token", "<unk>"))
        unk_token = str(payload.get("unk_token", "<unk>"))

        bos_token_id = tokenizer.token_to_id(bos_token)
        eos_token_id = tokenizer.token_to_id(eos_token)
        pad_token_id = tokenizer.token_to_id(pad_token)
        if bos_token_id is None or eos_token_id is None or pad_token_id is None:
            raise ValueError(
                "Step-Audio tokenizer.json is missing one of the required special tokens "
                f"({bos_token}, {eos_token}, {pad_token})."
            )

        return cls(
            tokenizer=tokenizer,
            model_dir=resolved_dir,
            tokenizer_path=tokenizer_path,
            tokenizer_config_path=tokenizer_config_path,
            chat_template=chat_template,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            unk_token=unk_token,
            bos_token_id=int(bos_token_id),
            eos_token_id=int(eos_token_id),
            pad_token_id=int(pad_token_id),
        )

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return [int(token_id) for token_id in encoding.ids]

    def decode(self, token_ids: Iterable[int], *, skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens)

    def token_to_id(self, token: str) -> int:
        token_id = self.tokenizer.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Tokenizer does not define token: {token}")
        return int(token_id)

    @staticmethod
    def render_messages(messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> str:
        if not messages:
            raise ValueError("messages must not be empty")

        rendered_parts: list[str] = []
        if messages[0]["role"] == "system":
            rendered_parts.append("<s>")

        for index, message in enumerate(messages):
            role = "human" if message["role"] == "user" else str(message["role"])
            content = message["content"]
            if isinstance(content, list):
                content = "".join(
                    str(item.get("text", ""))
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            rendered_parts.append(f"<|BOT|> {role}\n{content}")
            if index != len(messages) - 1 or message["role"] != "assistant":
                rendered_parts.append("<|EOT|>")

        if add_generation_prompt:
            rendered_parts.append("<|BOT|> assistant\n")

        return "".join(rendered_parts)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str | list[int]:
        rendered = self.render_messages(messages, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return self.encode(rendered)
        return rendered

    def build_clone_prompt_ids(
        self,
        *,
        speaker: str,
        prompt_text: str,
        prompt_wav_tokens: str,
        target_text: str,
    ) -> list[int]:
        return self.apply_chat_template(
            build_clone_messages(
                speaker=speaker,
                prompt_text=prompt_text,
                prompt_wav_tokens=prompt_wav_tokens,
                target_text=target_text,
            ),
            tokenize=True,
            add_generation_prompt=True,
        )

    def build_edit_prompt_ids(
        self,
        *,
        instruct_prefix: str,
        audio_token_str: str,
    ) -> list[int]:
        return self.apply_chat_template(
            build_edit_messages(
                instruct_prefix=instruct_prefix,
                audio_token_str=audio_token_str,
            ),
            tokenize=True,
            add_generation_prompt=True,
        )


__all__ = [
    "AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL",
    "AUDIO_EDIT_SYSTEM_PROMPT",
    "DEFAULT_STEP_AUDIO_CHAT_TEMPLATE",
    "StepAudioEditXTokenizer",
    "build_clone_messages",
    "build_edit_messages",
    "format_audio_token_string",
]
