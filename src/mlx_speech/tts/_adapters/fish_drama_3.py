"""Fish Audio Drama 3 TTS adapter.

Voice references follow upstream fish-speech: parallel entries of audio, transcript
and an optional speaker id (0..15). A transcript that does not start with a
<|speaker:N|> tag is tagged with its speaker; without one, the model default is
speaker 0. The text addresses speakers with the same tags.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.fish_drama_3 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    Drama3Reference,
    FishDrama3Runtime,
)


def _entries(value, count: int, name: str) -> list:
    """value as one entry per reference: a list of count, or one value for one."""
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(
                f"{name} needs one entry per reference; got {len(value)} for {count}"
            )
        return list(value)
    if count != 1 and value is not None:
        raise ValueError(f"{name} must be a list with one entry per reference")
    return [value] * count


class FishDrama3Adapter:
    def __init__(self, runtime: FishDrama3Runtime):
        self._runtime = runtime

    @classmethod
    def from_dir(cls, model_dir: Path) -> FishDrama3Adapter:
        return cls(FishDrama3Runtime.from_dir(model_dir))

    def prepare_reference(
        self,
        reference_audio: str | Path,
        *,
        reference_text: str,
        reference_speaker: int | None = None,
    ) -> Drama3Reference:
        """Encode a voice reference once; pass it back as reference_audio to reuse
        the voice across generate calls without encoding it again."""
        return self._reference(reference_audio, reference_text, reference_speaker, 1)

    def generate(
        self,
        text: str,
        *,
        reference_audio=None,
        reference_text=None,
        reference_speaker=None,
        duration_seconds: float | None = None,
        max_new_tokens: int | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        seed: int | None = None,
        single_chunk: bool = False,
        **kwargs,
    ) -> TTSOutput:
        """duration_seconds asks for audio of that length (a dubbing line's slot);
        max_new_tokens caps the frames of each text chunk (about 21.5 per second)."""
        references = self._references(
            reference_audio, reference_text, reference_speaker
        )
        chunks = list(
            self._runtime.synthesize(
                text,
                references,
                duration_seconds=duration_seconds,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                single_chunk=single_chunk,
                seed=seed,
            )
        )
        if not chunks:
            raise ValueError("Drama 3 stopped before generating any audio")
        return TTSOutput(
            waveform=mx.concatenate([chunk.waveform for chunk in chunks]),
            sample_rate=self._runtime.sample_rate,
        )

    def generate_batch(
        self,
        texts: Sequence[str],
        *,
        reference_audio=None,
        reference_text=None,
        reference_speaker=None,
        duration_seconds: float | Sequence[float | None] | None = None,
        max_new_tokens: int | Sequence[int | None] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        seed: int = 0,
        seeds: Sequence[int] | None = None,
        batch_size: int | None = None,
    ) -> list[TTSOutput]:
        """Generate independent lines together, in input order.

        The voice references are shared by every line, and each line is one pass,
        as with single_chunk. duration_seconds and max_new_tokens are one value or
        one per line. Seeds default to seed + line index; each line draws with its
        own seed, whatever the lines beside it. batch_size lines decode together, 128 by
        default. A warning names the lines that reached their frame budget before
        ending, whose speech may be cut off, and those that produced no audio.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string")
        references = self._references(
            reference_audio, reference_text, reference_speaker
        )
        lines = self._runtime.synthesize_batch(
            texts,
            references,
            seeds=list(seeds) if seeds is not None else [seed + i for i in range(len(texts))],
            duration_seconds=duration_seconds,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            batch_size=DEFAULT_BATCH_SIZE if batch_size is None else batch_size,
        )
        capped = [i for i, line in enumerate(lines) if line.finish_reason == "length"]
        silent = [i for i, line in enumerate(lines) if line.codes.shape[1] == 0]
        if capped:
            warnings.warn(
                f"Drama 3 lines {capped} reached their frame budget before ending; "
                "their speech may be cut off.",
                RuntimeWarning,
                stacklevel=2,
            )
        if silent:
            warnings.warn(
                f"Drama 3 lines {silent} stopped before generating any audio.",
                RuntimeWarning,
                stacklevel=2,
            )
        return [
            TTSOutput(waveform=line.waveform, sample_rate=self._runtime.sample_rate)
            for line in lines
        ]

    def _references(self, audio, text, speaker) -> list[Drama3Reference]:
        if audio is None:
            if text is not None or speaker is not None:
                raise ValueError(
                    "reference_text and reference_speaker need reference_audio"
                )
            return []
        audios = list(audio) if isinstance(audio, (list, tuple)) else [audio]
        texts = _entries(text, len(audios), "reference_text")
        speakers = _entries(speaker, len(audios), "reference_speaker")
        return [
            self._reference(*entry, index)
            for index, entry in enumerate(zip(audios, texts, speakers), 1)
        ]

    def _reference(self, audio, text, speaker, index: int) -> Drama3Reference:
        if isinstance(audio, Drama3Reference):
            if text is not None or speaker is not None:
                raise ValueError(
                    f"reference {index} is prepared and already carries its "
                    "transcript and speaker"
                )
            return audio
        if not isinstance(audio, (str, Path)):
            raise TypeError(
                f"reference {index}: expected an audio file path or a prepared "
                f"reference; got {type(audio).__name__}"
            )
        if text is None:
            raise ValueError(f"reference {index} needs its transcript")
        return self._runtime.encode_reference(audio, text, speaker=speaker)
