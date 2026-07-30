"""Tokenizer and deterministic text/audio generation schedules for dots.tts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


AUDIO_COMP_SPAN_TOKEN = "<|audio_comp_span|>"
AUDIO_GEN_START_TOKEN = "<|audio_gen_start|>"
AUDIO_GEN_SPAN_TOKEN = "<|audio_gen_span|>"
AUDIO_GEN_END_TOKEN = "<|audio_gen_end|>"
TEXT_COND_END_TOKEN = "<|text_cond_end|>"
TTS_TEXT_PREFIX = "[文本]"
TTS_AUDIO_PREFIX = "[文本对应语音]"
TTS_INTERLEAVE_PREFIX = "[流式语音合成]"


@dataclass(frozen=True)
class DotsTTSTokenizer:
    backend: Any
    audio_gen_start_id: int
    audio_gen_span_id: int
    audio_gen_end_id: int
    audio_comp_span_id: int
    text_cond_end_id: int

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "DotsTTSTokenizer":
        root = Path(model_dir)
        tokenizer_dir = root / "tokenizer" if (root / "tokenizer").is_dir() else root
        tokenizer_path = tokenizer_dir / "tokenizer.json"
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"dots.tts tokenizer is missing: {tokenizer_path}")
        try:
            from tokenizers import Tokenizer
        except ImportError as error:  # pragma: no cover - required project dependency
            raise ImportError("dots.tts tokenization requires tokenizers") from error
        try:
            backend = Tokenizer.from_file(str(tokenizer_path))
        except Exception as error:
            raise ValueError(f"invalid dots.tts tokenizer: {tokenizer_path}") from error
        ids = {
            token: backend.token_to_id(token)
            for token in (
                AUDIO_GEN_START_TOKEN,
                AUDIO_GEN_SPAN_TOKEN,
                AUDIO_GEN_END_TOKEN,
                AUDIO_COMP_SPAN_TOKEN,
                TEXT_COND_END_TOKEN,
            )
        }
        missing = sorted(token for token, token_id in ids.items() if token_id is None)
        if missing:
            raise ValueError(f"dots.tts tokenizer is missing required tokens: {missing}")
        return cls(
            backend=backend,
            audio_gen_start_id=int(ids[AUDIO_GEN_START_TOKEN]),
            audio_gen_span_id=int(ids[AUDIO_GEN_SPAN_TOKEN]),
            audio_gen_end_id=int(ids[AUDIO_GEN_END_TOKEN]),
            audio_comp_span_id=int(ids[AUDIO_COMP_SPAN_TOKEN]),
            text_cond_end_id=int(ids[TEXT_COND_END_TOKEN]),
        )

    def encode(self, text: str) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=False).ids)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        return str(
            self.backend.decode(token_ids, skip_special_tokens=skip_special_tokens)
        )


@dataclass(frozen=True)
class DotsTTSSchedule:
    token_ids: tuple[int, ...]
    text_token_count: int
    audio_span_positions: tuple[int, ...]
    interleave: bool

    @property
    def audio_patch_budget(self) -> int:
        return len(self.audio_span_positions)


def normalize_language_code(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().upper().replace("_", "-")
    if normalized in {"", "NONE"}:
        return None
    primary = normalized.split("-", 1)[0]
    if not (2 <= len(primary) <= 3 and primary.isalpha()):
        raise ValueError(f"invalid dots.tts language code: {language!r}")
    return "口音:粤语" if primary == "YUE" else primary


def attach_language_tag(text: str, language: str | None) -> str:
    stripped = text.strip()
    if not stripped:
        raise ValueError("dots.tts text must not be empty")
    language_code = normalize_language_code(language)
    if language_code is None:
        return stripped
    tag = f"[{language_code}]"
    return stripped if stripped.startswith(tag) else f"{tag}{stripped}"


def prepare_conditioned_text(
    text: str,
    *,
    language: str | None,
    prompt_text: str | None = None,
) -> tuple[str, str]:
    target = text.strip()
    if not target:
        raise ValueError("dots.tts target text must not be empty")
    prompt = "" if prompt_text is None else prompt_text.strip()
    if prompt:
        prompt = attach_language_tag(prompt, language) + "\n"
    else:
        target = attach_language_tag(target, language)
    return prompt, target


def build_generation_schedule(
    *,
    text: str,
    tokenizer: DotsTTSTokenizer,
    max_audio_patches: int,
    template: Literal["tts", "tts_interleave"] = "tts",
) -> DotsTTSSchedule:
    if max_audio_patches <= 0:
        raise ValueError("max_audio_patches must be positive")
    text_ids = tokenizer.encode(text)
    if not text_ids:
        raise ValueError("dots.tts text produced no tokens")

    if template == "tts":
        schedule = [
            *tokenizer.encode(TTS_TEXT_PREFIX),
            *text_ids,
            *tokenizer.encode(TTS_AUDIO_PREFIX),
            tokenizer.audio_gen_start_id,
            *([tokenizer.audio_gen_span_id] * max_audio_patches),
        ]
        interleave = False
    elif template == "tts_interleave":
        if max_audio_patches < len(text_ids):
            raise ValueError(
                "interleave requires at least one audio patch per text token: "
                f"text_tokens={len(text_ids)}, max_audio_patches={max_audio_patches}"
            )
        schedule = tokenizer.encode(TTS_INTERLEAVE_PREFIX)
        for token_id in text_ids:
            schedule.extend((token_id, tokenizer.audio_gen_span_id))
        schedule.append(tokenizer.text_cond_end_id)
        schedule.extend(
            [tokenizer.audio_gen_span_id] * (max_audio_patches - len(text_ids))
        )
        interleave = True
    else:
        raise ValueError(f"unsupported dots.tts generation template: {template}")

    positions = tuple(
        index
        for index, token_id in enumerate(schedule)
        if token_id == tokenizer.audio_gen_span_id
    )
    return DotsTTSSchedule(
        token_ids=tuple(schedule),
        text_token_count=len(text_ids),
        audio_span_positions=positions,
        interleave=interleave,
    )


__all__ = [
    "DotsTTSSchedule",
    "DotsTTSTokenizer",
    "attach_language_tag",
    "build_generation_schedule",
    "normalize_language_code",
    "prepare_conditioned_text",
]
