"""R2T2 streaming state machine.

The Qwen graph stays outside this module. A caller injects ``llm_generate``
for one completion step and ``tokenize`` / ``detokenize`` for prefix rollback.
"""

from __future__ import annotations

import re
import string
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..qwen3_asr.processor import resolve_language
from .metrics import R2T2StepMetrics, R2T2StreamTrace


SAMPLE_RATE = 16000
_BUDGET_SAMPLES = 1280
_PCM_ITEMSIZE = np.dtype(np.float32).itemsize
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "
_PREFIX_PUNCT = "，。！？、；：.!?;:"
_FIXED_PUNCT = "，。！？、；：,.!?;:"
_EN2ZH_PUNCT = {
    ",": "，",
    ".": "。",
    "!": "！",
    "?": "？",
    ";": "；",
    ":": "：",
    "(": "（",
    ")": "）",
}
_ZH2EN_PUNCT = {value: key for key, value in _EN2ZH_PUNCT.items()}
_ALL_PUNCT_PAT = re.compile(
    r"[,\.!?;:()\uff0c\u3002\uff01\uff1f\uff1b\uff1a\uff08\uff09]"
)
_CJK_SPACE_PAT = re.compile(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])")

LlmGenerate = Callable[[str, np.ndarray, int], str]
Tokenize = Callable[[str], Sequence[int]]
Detokenize = Callable[[Sequence[int]], str]


@dataclass(frozen=True)
class R2T2StreamUpdate:
    """One ``feed`` result. ``committed`` is append-only; ``hypothesis`` may change."""

    delta: str
    committed: str
    hypothesis: str
    language: str


@dataclass(frozen=True)
class R2T2Transcript:
    """Final committed transcript after ``finalize``."""

    text: str
    language: str


def _normalize_punct_by_context(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        punct = match.group()
        prev_char = ""
        for index in range(match.start() - 1, -1, -1):
            if not text[index].isspace():
                prev_char = text[index]
                break
        if not prev_char:
            return punct
        if "\u4e00" <= prev_char <= "\u9fff":
            return _EN2ZH_PUNCT.get(punct, punct)
        if prev_char.isascii() and (prev_char.isalnum() or prev_char in "\"'"):
            return _ZH2EN_PUNCT.get(punct, punct)
        return punct

    return _ALL_PUNCT_PAT.sub(_replace, text)


def _detect_and_fix_repetitions(text: str, threshold: int = 20) -> str:
    def fix_char_repeats(value: str, thresh: int) -> str:
        parts: list[str] = []
        index = 0
        size = len(value)
        while index < size:
            count = 1
            while index + count < size and value[index + count] == value[index]:
                count += 1
            if count > thresh:
                parts.append(value[index])
                index += count
            else:
                parts.append(value[index : index + count])
                index += count
        return "".join(parts)

    def fix_pattern_repeats(value: str, thresh: int, max_len: int = 20) -> str:
        size = len(value)
        min_repeat_chars = thresh * 2
        if size < min_repeat_chars:
            return value
        index = 0
        parts: list[str] = []
        found = False
        while index <= size - min_repeat_chars:
            found = False
            for width in range(1, max_len + 1):
                if index + width * thresh > size:
                    break
                pattern = value[index : index + width]
                valid = True
                for rep in range(1, thresh):
                    start = index + rep * width
                    if value[start : start + width] != pattern:
                        valid = False
                        break
                if valid:
                    end_index = index + thresh * width
                    while (
                        end_index + width <= size
                        and value[end_index : end_index + width] == pattern
                    ):
                        end_index += width
                    parts.append(pattern)
                    parts.append(
                        fix_pattern_repeats(value[end_index:], thresh, max_len)
                    )
                    index = size
                    found = True
                    break
            if found:
                break
            parts.append(value[index])
            index += 1
        if not found:
            parts.append(value[index:])
        return "".join(parts)

    return fix_pattern_repeats(fix_char_repeats(text, threshold), threshold)


def _normalize_language_token(language: str) -> str:
    value = str(language).strip()
    if not value:
        return ""
    return value[:1].upper() + value[1:].lower()


def parse_language_output(
    raw: str | None, user_language: str | None = None
) -> tuple[str, str]:
    """Language probe for the CJK-space decision. No repetition fixer."""
    if raw is None:
        return "", ""
    if user_language == "English":
        text = str(raw).rstrip()
    else:
        text = str(raw).strip()
    if not text:
        return "", ""
    if user_language:
        return user_language, text
    if _ASR_TEXT_TAG not in text:
        return "", text.strip()

    meta_part, text_part = text.split(_ASR_TEXT_TAG, 1)
    if "language none" in meta_part.lower():
        body = text_part.strip()
        return ("", "") if not body else ("", body)

    language = ""
    for line in meta_part.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(_LANG_PREFIX):
            value = stripped[len(_LANG_PREFIX) :].strip()
            if value:
                language = _normalize_language_token(value)
            break
    return language, text_part.strip()


def parse_streaming_asr_output(
    raw: str | None, user_language: str | None = None
) -> tuple[str, str]:
    """Qwen streaming parse: strip, fix repetitions, then split ``<asr_text>``.

    Not ``mlx_speech.models.qwen3_asr.processor.parse_asr_output``. With no tag
    this returns ``("", text)`` and does not infer a language.
    """
    if raw is None:
        return "", ""
    text = str(raw).strip()
    if not text:
        return "", ""
    text = _detect_and_fix_repetitions(text)
    if user_language:
        return user_language, text
    if _ASR_TEXT_TAG not in text:
        return "", text.strip()

    meta_part, text_part = text.split(_ASR_TEXT_TAG, 1)
    if "language none" in meta_part.lower():
        body = text_part.strip()
        return ("", "") if not body else ("", body)

    language = ""
    for line in meta_part.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith(_LANG_PREFIX):
            value = stripped[len(_LANG_PREFIX) :].strip()
            if value:
                language = _normalize_language_token(value)
            break
    return language, text_part.strip()


def split_text_to_tokens(text: str) -> list[str]:
    """CJK is one character, Latin is a word, punctuation is dropped."""
    cleaned = text.translate(_punctuation_table())
    tokens: list[str] = []
    for match in re.finditer(r"[a-zA-Z]+|[^a-zA-Z]", cleaned):
        token = match.group()
        if token.strip():
            tokens.append(token)
    return tokens


def is_last_token_chinese(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    return any("\u4e00" <= char <= "\u9fff" for char in tokens[-1])


def _punctuation_table() -> dict[int, int | None]:
    chinese = "？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
    return str.maketrans("", "", string.punctuation + chinese)


def _ms_to_samples(ms: int) -> int:
    return int(round(float(ms) / 1000.0 * SAMPLE_RATE))


def _require_chunk_ms(chunk_ms: int) -> int:
    if isinstance(chunk_ms, bool) or not isinstance(chunk_ms, int):
        raise TypeError("chunk_ms must be an int.")
    if chunk_ms < 80 or chunk_ms > 2000:
        raise ValueError("chunk_ms must be in [80, 2000].")
    return chunk_ms


def _rollback_k(raw: str, unfixed_token_num: int, punctuation: str | None) -> int:
    if punctuation is not None:
        stripped = raw.strip()
        if stripped and stripped[-1] in punctuation:
            return 0
    return int(unfixed_token_num)


def _detokenize_prefix(
    raw: str,
    *,
    tokenize: Tokenize,
    detokenize: Detokenize,
    k: int,
    min_end: int,
    drop_replacement: bool,
) -> str:
    ids = list(tokenize(raw))
    while True:
        end_idx = max(min_end, len(ids) - k)
        text = detokenize(ids[:end_idx]) if end_idx > 0 else ""
        if not drop_replacement or "\ufffd" not in text:
            return text
        if end_idx == 0:
            return ""
        k += 1


class R2T2StreamSession:
    """Published R2T2 chunk loop, lookahead, budget, and append-only commit.

    ``trace`` is opt-in profiling: with ``None`` the loop keeps no
    per-step records and does no timing work.
    """

    def __init__(
        self,
        llm_generate: LlmGenerate,
        tokenize: Tokenize,
        detokenize: Detokenize,
        *,
        language: str | None = None,
        chunk_ms: int = 160,
        lookahead_ms: int = 160,
        unfixed_token_num: int = 1,
        unfixed_chunk_num: int = 0,
        rollback_punctuation: bool = False,
        trace: R2T2StreamTrace | None = None,
    ) -> None:
        if (
            not callable(llm_generate)
            or not callable(tokenize)
            or not callable(detokenize)
        ):
            raise TypeError("llm_generate, tokenize, and detokenize must be callable.")
        chunk_ms = _require_chunk_ms(chunk_ms)
        if (
            isinstance(lookahead_ms, bool)
            or not isinstance(lookahead_ms, int)
            or lookahead_ms < 0
        ):
            raise TypeError("lookahead_ms must be a non-negative int.")
        if unfixed_token_num < 0 or unfixed_chunk_num < 0:
            raise ValueError("unfixed_token_num and unfixed_chunk_num must be >= 0.")

        self._llm_generate = llm_generate
        self._tokenize = tokenize
        self._detokenize = detokenize
        self._force_language = resolve_language(language)
        self._chunk_samples = max(1, _ms_to_samples(chunk_ms))
        self._lookahead_samples = _ms_to_samples(lookahead_ms)
        self._unfixed_token_num = int(unfixed_token_num)
        self._unfixed_chunk_num = int(unfixed_chunk_num)
        self._rollback_punctuation = bool(rollback_punctuation)
        self._trace = trace

        self._buffer = np.zeros((0,), dtype=np.float32)
        self._audio_store = np.zeros((0,), dtype=np.float32)
        self._audio_len = 0
        self._awaiting_first = True
        self.chunk_id = 0
        self._raw_decoded = ""
        self._text = ""
        self._language = ""
        self._committed = ""
        self._committed_tokens: list[str] = []
        first_samples = self._chunk_samples + self._lookahead_samples
        self._max_new_tokens = float(max(1, first_samples // _BUDGET_SAMPLES))
        self._floor = float(
            min(32, max(4, 2 * (self._chunk_samples // _BUDGET_SAMPLES)))
        )
        self._finalize_cap = int(self._max_new_tokens)

    @property
    def language(self) -> str | None:
        return self._force_language

    @property
    def audio_samples(self) -> int:
        return int(self._audio_accum.shape[0])

    @property
    def trace(self) -> R2T2StreamTrace | None:
        return self._trace

    def feed(self, pcm16k: np.ndarray) -> R2T2StreamUpdate:
        incoming = _as_float_pcm(pcm16k)
        if incoming.shape[0] > 0:
            self._note_copy(
                int(self._buffer.shape[0]) + int(incoming.shape[0]),
                defer=True,
            )
            self._buffer = np.concatenate([self._buffer, incoming])
        deltas: list[str] = []
        while self._buffer.shape[0] >= self._current_window():
            window = self._current_window()
            chunk = self._buffer[:window]
            self._buffer = self._buffer[window:]
            self._awaiting_first = False
            metrics = self._begin_step(kind="window")
            started = time.perf_counter()
            self._append_audio(chunk)
            budget = int(self._max_new_tokens)
            if metrics is not None:
                metrics.budget = budget
            text, fixed_text = self._streaming_step(budget)
            delta = self._lock(fixed_text)
            deltas.append(delta)
            self._update_budget(delta)
            self._text = text
            self._end_step(metrics, hypothesis=text, delta=delta, started=started)
        return self._update("".join(deltas))

    def finalize(self) -> R2T2Transcript:
        if self._buffer.shape[0] == 0:
            self._lock(self._text)
            return R2T2Transcript(text=self._committed, language=self._language)

        tail = self._buffer
        self._buffer = np.zeros((0,), dtype=np.float32)
        metrics = self._begin_step(kind="tail")
        started = time.perf_counter()
        self._append_audio(tail)
        if metrics is not None:
            metrics.budget = int(self._finalize_cap)
        self._text = self._finish_step(self._finalize_cap)
        self._lock(self._text)
        self._end_step(metrics, hypothesis=self._text, delta="", started=started)
        return R2T2Transcript(text=self._committed, language=self._language)

    def _current_window(self) -> int:
        if self._awaiting_first:
            return self._chunk_samples + self._lookahead_samples
        return self._chunk_samples

    @property
    def _audio_accum(self) -> np.ndarray:
        """Read-only view of all audio so far. Written samples never change,
        so a view handed to the decoder stays valid after later appends."""

        view = self._audio_store[: self._audio_len]
        view.flags.writeable = False
        return view

    def _append_audio(self, chunk: np.ndarray) -> None:
        chunk_samples = int(chunk.shape[0])
        end = self._audio_len + chunk_samples
        if end > self._audio_store.shape[0]:
            capacity = max(end, 2 * int(self._audio_store.shape[0]), 16 * 16_000)
            grown = np.zeros((capacity,), dtype=np.float32)
            grown[: self._audio_len] = self._audio_store[: self._audio_len]
            self._note_copy(self._audio_len)
            self._audio_store = grown
        self._audio_store[self._audio_len : end] = chunk
        self._note_copy(chunk_samples)
        self._audio_len = end

    def _begin_step(self, *, kind: str) -> R2T2StepMetrics | None:
        if self._trace is None:
            return None
        position = int(self._audio_accum.shape[0])
        return self._trace.begin(
            index=len(self._trace.steps),
            kind=kind,
            sample_start=position,
            sample_end=position,
        )

    def _end_step(
        self,
        metrics: R2T2StepMetrics | None,
        *,
        hypothesis: str,
        delta: str,
        started: float,
    ) -> None:
        if self._trace is None or metrics is None:
            return
        metrics.sample_end = int(self._audio_accum.shape[0])
        metrics.hypothesis = hypothesis
        metrics.delta = delta
        metrics.committed = self._committed
        metrics.step_seconds = time.perf_counter() - started
        self._trace.end()

    def _note_copy(self, samples: int, *, defer: bool = False) -> None:
        if self._trace is not None:
            self._trace.count_copy(samples, _PCM_ITEMSIZE, defer=defer)

    def _generate(self, prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
        trace = self._trace
        if trace is None:
            return self._llm_generate(prefix, audio, max_new_tokens)
        started = time.perf_counter()
        try:
            return self._llm_generate(prefix, audio, max_new_tokens)
        finally:
            if trace.current is not None:
                trace.current.llm_seconds += time.perf_counter() - started

    def _streaming_step(self, max_new_tokens: int) -> tuple[str, str]:
        if self.chunk_id < self._unfixed_chunk_num:
            prefix = ""
        else:
            self._raw_decoded = self._raw_decoded.split("|", 1)[0]
            punctuation = _PREFIX_PUNCT if self._rollback_punctuation else None
            k = _rollback_k(self._raw_decoded, self._unfixed_token_num, punctuation)
            prefix = _detokenize_prefix(
                self._raw_decoded,
                tokenize=self._tokenize,
                detokenize=self._detokenize,
                k=k,
                min_end=0,
                drop_replacement=True,
            )
        prefix = prefix.split("|", 1)[0]
        audio_copy = self._audio_accum
        gen_text = self._generate(prefix, audio_copy, max_new_tokens)
        gen_text = _normalize_punct_by_context(gen_text).replace("\ufffd", "")
        self._raw_decoded = prefix + gen_text

        probed = None
        if self._force_language is None:
            probed, _ = parse_language_output(self._raw_decoded, user_language=None)
        if self._force_language == "Chinese" or probed == "Chinese":
            self._raw_decoded = _CJK_SPACE_PAT.sub("", self._raw_decoded)

        language, text = parse_streaming_asr_output(
            self._raw_decoded,
            user_language=self._force_language,
        )
        if _ASR_TEXT_TAG in self._raw_decoded:
            meta, _body = self._raw_decoded.split(_ASR_TEXT_TAG, 1)
            self._raw_decoded = meta + _ASR_TEXT_TAG + text
        else:
            self._raw_decoded = text
        self._raw_decoded = self._raw_decoded.split("|", 1)[0]

        punctuation = _FIXED_PUNCT if self._rollback_punctuation else None
        k = _rollback_k(self._raw_decoded, self._unfixed_token_num, punctuation)
        if (
            _ASR_TEXT_TAG in self._raw_decoded
            and self._raw_decoded.split(_ASR_TEXT_TAG, 1)[1] == ""
        ):
            k = 0
        fixed_text = _detokenize_prefix(
            self._raw_decoded,
            tokenize=self._tokenize,
            detokenize=self._detokenize,
            k=k,
            min_end=0,
            drop_replacement=True,
        )
        if _ASR_TEXT_TAG in fixed_text:
            _meta, fixed_text = fixed_text.split(_ASR_TEXT_TAG, 1)
        fixed_text = fixed_text.split("|", 1)[0]

        if _ASR_TEXT_TAG not in self._raw_decoded and self._force_language is None:
            return "", ""

        self._language = language
        self.chunk_id += 1
        return text.split("|", 1)[0], fixed_text

    def _finish_step(self, max_new_tokens: int) -> str:
        if self.chunk_id < self._unfixed_chunk_num:
            prefix = ""
        else:
            ids = list(self._tokenize(self._raw_decoded))
            end_idx = max(1, len(ids) - self._unfixed_token_num)
            prefix = self._detokenize(ids[:end_idx])
        prefix = prefix.split("|", 1)[0]
        audio_copy = self._audio_accum
        gen_text = self._generate(prefix, audio_copy, max_new_tokens)
        gen_text = _normalize_punct_by_context(gen_text).replace("\ufffd", "")
        self._raw_decoded = (prefix + gen_text).split("|", 1)[0]
        language, text = parse_streaming_asr_output(
            self._raw_decoded,
            user_language=self._force_language,
        )
        self._language = language
        self.chunk_id += 1
        return text.split("|", 1)[0]

    def _lock(self, candidate: str) -> str:
        if len(candidate) > len(self._committed):
            delta = candidate[len(self._committed) :]
            self._committed = self._committed + delta
            return delta
        return ""

    def _update_budget(self, delta: str) -> None:
        base = max(1, self._chunk_samples // _BUDGET_SAMPLES)
        if delta:
            self._committed_tokens.extend(split_text_to_tokens(delta))
            self._max_new_tokens = float(base)
        elif not is_last_token_chinese(self._committed_tokens):
            self._max_new_tokens += 0.5
        else:
            self._max_new_tokens = float(base)
        if is_last_token_chinese(self._committed_tokens):
            self._max_new_tokens *= 2
        self._max_new_tokens = min(self._floor, self._max_new_tokens)

    def _update(self, delta: str) -> R2T2StreamUpdate:
        return R2T2StreamUpdate(
            delta=delta,
            committed=self._committed,
            hypothesis=self._text,
            language=self._language,
        )


def _as_float_pcm(pcm16k: np.ndarray) -> np.ndarray:
    audio = np.asarray(pcm16k)
    if audio.ndim != 1:
        audio = audio.reshape(-1)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / np.float32(32768.0)
    return audio.astype(np.float32, copy=False)
