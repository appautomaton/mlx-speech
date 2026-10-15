"""Fish Audio Drama 3 TTS adapter.

Voice references follow upstream fish-speech: parallel entries of audio, transcript
and an optional speaker id (0..15). A transcript that does not start with a
<|speaker:N|> tag is tagged with its speaker; without one, the model default is
speaker 0. The text addresses speakers with the same tags.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from .._adapter import TTSOutput
from ...generation.fish_drama_3 import (
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
        if seed is not None:
            mx.random.seed(seed)
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
            )
        )
        if not chunks:
            raise ValueError("Drama 3 stopped before generating any audio")
        return TTSOutput(
            waveform=mx.concatenate([chunk.waveform for chunk in chunks]),
            sample_rate=self._runtime.sample_rate,
        )

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
