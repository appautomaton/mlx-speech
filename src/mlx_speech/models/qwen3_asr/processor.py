"""Prompt construction and output parsing for Qwen3-ASR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import Qwen3ASRConfig
from .feature_extraction import Qwen3ASRFeatureExtractor
from .tokenizer import Qwen3ASRTokenizer


ASR_TEXT_TAG = "<asr_text>"
LANG_PREFIX = "language "
SUPPORTED_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "Arabic",
    "German",
    "French",
    "Spanish",
    "Portuguese",
    "Indonesian",
    "Italian",
    "Korean",
    "Russian",
    "Thai",
    "Vietnamese",
    "Japanese",
    "Turkish",
    "Hindi",
    "Malay",
    "Dutch",
    "Swedish",
    "Danish",
    "Finnish",
    "Polish",
    "Czech",
    "Filipino",
    "Persian",
    "Greek",
    "Romanian",
    "Hungarian",
    "Macedonian",
)


@dataclass(frozen=True)
class Qwen3ASRPrompt:
    """One tokenized Qwen3-ASR prompt."""

    prompt: str
    input_ids: list[int]
    language: str | None
    audio_length: int


@dataclass(frozen=True)
class Qwen3ASRProcessorOutput:
    """Processor output consumed by the future Qwen3-ASR runtime."""

    input_ids: list[list[int]]
    input_features: np.ndarray
    feature_attention_mask: np.ndarray
    audio_lengths: np.ndarray
    prompts: tuple[str, ...]
    languages: tuple[str | None, ...]


class Qwen3ASRProcessor:
    """Pure local equivalent of Qwen3-ASR prompt/audio processor behavior."""

    def __init__(
        self,
        *,
        config: Qwen3ASRConfig,
        tokenizer: Qwen3ASRTokenizer,
        feature_extractor: Qwen3ASRFeatureExtractor,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "Qwen3ASRProcessor":
        model_dir = Path(model_dir)
        return cls(
            config=Qwen3ASRConfig.from_dir(model_dir),
            tokenizer=Qwen3ASRTokenizer.from_dir(model_dir),
            feature_extractor=Qwen3ASRFeatureExtractor.from_dir(model_dir),
        )

    @property
    def supported_languages(self) -> tuple[str, ...]:
        configured = tuple(getattr(self.config, "support_languages", ()) or ())
        return configured or SUPPORTED_LANGUAGES

    def resolve_language(self, language: str | None) -> str | None:
        """Return canonical forced language, or None for auto detection."""

        return resolve_language(language, supported_languages=self.supported_languages)

    def build_prompt(
        self,
        *,
        context: str | None = "",
        audio_length: int,
        language: str | None = None,
    ) -> Qwen3ASRPrompt:
        audio_length = int(audio_length)
        if audio_length < 0:
            raise ValueError("audio_length must be non-negative.")

        forced_language = self.resolve_language(language)
        audio_span = (
            self.tokenizer.audio_bos_token
            + (self.tokenizer.audio_token * audio_length)
            + self.tokenizer.audio_eos_token
        )
        prompt = (
            f"<|im_start|>system\n{context or ''}<|im_end|>\n"
            f"<|im_start|>user\n{audio_span}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        if forced_language is not None:
            prompt += f"{LANG_PREFIX}{forced_language}{ASR_TEXT_TAG}"

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        observed_audio_tokens = sum(
            1 for token_id in input_ids if int(token_id) == self.tokenizer.audio_token_id
        )
        if observed_audio_tokens != audio_length:
            raise ValueError(
                "Qwen3-ASR prompt audio token count mismatch: "
                f"expected {audio_length}, got {observed_audio_tokens}."
            )
        return Qwen3ASRPrompt(
            prompt=prompt,
            input_ids=[int(token_id) for token_id in input_ids],
            language=forced_language,
            audio_length=audio_length,
        )

    def build_batch_prompts(
        self,
        *,
        audio_lengths: Sequence[int] | np.ndarray,
        contexts: str | Sequence[str | None] | None = "",
        languages: str | Sequence[str | None] | None = None,
    ) -> tuple[Qwen3ASRPrompt, ...]:
        lengths = [int(x) for x in np.asarray(audio_lengths, dtype=np.int64).reshape(-1)]
        context_items = _broadcast(contexts, len(lengths), name="contexts", default="")
        language_items = _broadcast(languages, len(lengths), name="languages", default=None)
        return tuple(
            self.build_prompt(
                context=context,
                audio_length=audio_length,
                language=language,
            )
            for context, audio_length, language in zip(
                context_items,
                lengths,
                language_items,
                strict=True,
            )
        )

    def __call__(
        self,
        audio: Any,
        *,
        context: str | Sequence[str | None] | None = "",
        language: str | Sequence[str | None] | None = None,
        sample_rate: int = 16000,
    ) -> Qwen3ASRProcessorOutput:
        feature_batch = self.feature_extractor(audio, sample_rate=sample_rate)
        audio_lengths = np.asarray(feature_batch.audio_lengths, dtype=np.int64)
        prompts = self.build_batch_prompts(
            audio_lengths=audio_lengths,
            contexts=context,
            languages=language,
        )
        return Qwen3ASRProcessorOutput(
            input_ids=[prompt.input_ids for prompt in prompts],
            input_features=feature_batch.input_features,
            feature_attention_mask=feature_batch.feature_attention_mask,
            audio_lengths=audio_lengths,
            prompts=tuple(prompt.prompt for prompt in prompts),
            languages=tuple(prompt.language for prompt in prompts),
        )


def normalize_language_name(language: str) -> str:
    if language is None:
        raise ValueError("language is None.")
    value = str(language).strip()
    if not value:
        raise ValueError("language is empty.")
    return value[:1].upper() + value[1:].lower()


def resolve_language(
    language: str | None,
    *,
    supported_languages: Sequence[str] = SUPPORTED_LANGUAGES,
) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    if not value or value.lower() == "auto":
        return None

    normalized = normalize_language_name(value)
    supported = tuple(supported_languages or SUPPORTED_LANGUAGES)
    if normalized not in supported:
        raise ValueError(f"Unsupported language: {normalized}. Supported: {list(supported)}")
    return normalized


def parse_asr_output(
    raw: str | None,
    *,
    user_language: str | None = None,
) -> tuple[str, str]:
    if raw is None:
        return "", ""
    text = str(raw).strip()
    if not text:
        return "", ""

    forced_language = resolve_language(user_language) if user_language else None
    if forced_language is not None:
        return forced_language, text

    if ASR_TEXT_TAG not in text:
        return "", text

    meta_part, text_part = text.split(ASR_TEXT_TAG, 1)
    transcript = text_part.strip()
    if "language none" in meta_part.lower():
        return "", transcript

    language = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.lower().startswith(LANG_PREFIX):
            value = line[len(LANG_PREFIX) :].strip()
            if value:
                language = normalize_language_name(value)
            break
    return language, transcript


def _broadcast(
    value: str | Sequence[str | None] | None,
    count: int,
    *,
    name: str,
    default: str | None,
) -> list[str | None]:
    if isinstance(value, str):
        return [value] * count
    if value is None:
        return [default] * count
    if not isinstance(value, Sequence):
        return [str(value)] * count

    items = list(value)
    if len(items) != count:
        raise ValueError(f"{name} length must match audio batch size {count}; got {len(items)}.")
    return [str(item) if item is not None else default for item in items]
