"""FireRedTTS3 text-token construction without Transformers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Japanese",
    "Korean",
    "Spanish",
    "French",
    "Russian",
    "Arabic",
    "Turkish",
    "Indonesian",
    "Portuguese",
    "Italian",
    "Dutch",
    "Vietnamese",
    "German",
    "Ukrainian",
    "Thai",
    "Polish",
    "Romanian",
    "Greek",
    "Czech",
    "Finnish",
    "Hindi",
)

DIALECTS = (
    "ZH_Anhui",
    "ZH_Fujian",
    "ZH_Gansu",
    "ZH_Guizhou",
    "ZH_Hebei",
    "ZH_Henan",
    "ZH_Hubei",
    "ZH_Hunan",
    "ZH_Jiangxi",
    "ZH_Liaoning",
    "ZH_Minnan",
    "ZH_Ningxia",
    "ZH_Shaanxi",
    "ZH_Shandong",
    "ZH_Shanghai",
    "ZH_Shanxi",
    "ZH_Sichuan",
    "ZH_Tianjin",
    "ZH_Wenzhou",
    "ZH_Wu",
    "ZH_Yunnan",
)

_TASK_TOKENS = (
    "<|sosp|>",
    "<|eosp|>",
    "<|empty|>",
    "<|Human|>",
    "<|SpeechLM|>",
    "<|sostm|>",
    "<|eostm|>",
    "<|sot|>",
    "<|eot|>",
    "<|TEXT_ONLY|>",
    "<|AUDIO_ONLY|>",
    "<|ASR|>",
    "<|TTS|>",
    "<|INTERLEAVE|>",
    "<|UNDERSTANDING|>",
)


def _official_added_tokens() -> list[str]:
    return [
        *_TASK_TOKENS,
        *(f"<|placeholder_{index:03d}|>" for index in range(1, 193)),
        *(f"<|{language}|>" for language in LANGUAGES),
        *(f"<|{dialect}|>" for dialect in DIALECTS),
        "<|edit|>",
        "<|frame_patch|>",
        "<|end_edit|>",
    ]


class FireRedTTS3Tokenizer:
    def __init__(self, backend: Any):
        self.backend = backend
        self.backend.add_special_tokens(_official_added_tokens())

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "FireRedTTS3Tokenizer":
        from tokenizers import Tokenizer

        path = Path(model_dir) / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"FireRedTTS3 tokenizer not found: {path}")
        return cls(Tokenizer.from_file(str(path)))

    @staticmethod
    def build_input(
        *,
        language: str,
        reference_text: str,
        text: str,
    ) -> str:
        if language not in (*LANGUAGES, *DIALECTS):
            raise ValueError(f"unsupported FireRedTTS3 language: {language}")
        return f"<|{language}|><|sot|>{reference_text}{text}<|eot|>"

    def encode(
        self,
        *,
        language: str,
        reference_text: str,
        text: str,
    ) -> list[int]:
        source = self.build_input(
            language=language,
            reference_text=reference_text,
            text=text,
        )
        return list(self.backend.encode(source, add_special_tokens=False).ids)


__all__ = ["DIALECTS", "LANGUAGES", "FireRedTTS3Tokenizer"]
