"""Fish Audio Drama 3 inference: text and voice references to a 44.1 kHz waveform.

Ported from upstream fish_omni.generate (one request, one chunk at a time) and the
chunked long-form loop of fish-speech's generate_long_drama3.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from ..audio import load_audio
from ..models._cache import BoundedKVCache
from ..models.fish_drama_3.chunking import ChunkHistory, split_text
from ..models.fish_drama_3.config import FishDrama3Config
from ..models.fish_drama_3.layers import DeltaNetState
from ..models.fish_drama_3.model import FishDrama3Model, load_fish_drama_3_model
from ..models.fish_drama_3.prompt import (
    MAX_SPEAKER_ID,
    PromptTooLongError,
    build_prompt,
    leads_with_speaker_tag,
    with_duration_hint,
)
from ..models.fish_drama_3.sampling import RAS_WINDOW_FRAMES
from ..models.fish_s2_pro.codec import FishS2Codec

# Upstream's reference recipe.
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.7
DEFAULT_TOP_K = 30
# Frames per chunk, about 47 s of audio.
DEFAULT_MAX_NEW_TOKENS = 1024
# A take may run to about 2.5 times its expected duration (technical report, 3.3).
RUNAWAY_FACTOR = 2.5
PREFILL_CHUNK_ROWS = 128

FinishReason = Literal["eos", "length"]


@dataclass(frozen=True)
class Drama3Reference:
    """A voice reference encoded once: codec codes plus its transcript."""

    codes: np.ndarray  # [num_codebooks, frames]
    text: str


@dataclass(frozen=True)
class Drama3Chunk:
    """The audio of one generated text chunk."""

    text: str
    codes: np.ndarray  # [num_codebooks, frames]
    waveform: mx.array
    finish_reason: FinishReason


def _cache_arrays(caches: Sequence[BoundedKVCache | DeltaNetState]) -> list[mx.array]:
    arrays = []
    for cache in caches:
        if isinstance(cache, DeltaNetState):
            arrays += [cache.conv, cache.recurrent]
        else:
            arrays += [cache.keys, cache.values]
    return arrays


@dataclass
class FishDrama3Runtime:
    model: FishDrama3Model
    tokenizer: Tokenizer
    codec: FishS2Codec

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> FishDrama3Runtime:
        resolved = Path(model_dir)
        return cls(
            model=load_fish_drama_3_model(resolved),
            tokenizer=Tokenizer.from_file(str(resolved / "tokenizer.json")),
            # Drama 3 ships the S2 Pro codec, byte for byte.
            codec=FishS2Codec.from_dir(resolved / "codec-mlx"),
        )

    @property
    def config(self) -> FishDrama3Config:
        return self.model.config

    @property
    def sample_rate(self) -> int:
        return self.codec.sample_rate

    @property
    def frame_rate(self) -> float:
        """Codec frames per second of audio."""
        return self.sample_rate / self.codec.model.model.frame_length

    def encode_reference(
        self, audio: str | Path, text: str, *, speaker: int | None = None
    ) -> Drama3Reference:
        """Encode a voice reference once for reuse.

        A transcript that does not start with a <|speaker:N|> tag gets speaker's tag;
        with speaker None it keeps the model default, speaker 0.
        """
        if speaker is not None and not 0 <= speaker <= MAX_SPEAKER_ID:
            raise ValueError(f"speaker must be 0..{MAX_SPEAKER_ID}; got {speaker}")
        waveform, _ = load_audio(audio, sample_rate=self.sample_rate, mono=True)
        codes = np.array(self.codec.encode(waveform), dtype=np.int32)
        if speaker is not None and not leads_with_speaker_tag(text):
            text = f"<|speaker:{speaker}|> {text}"
        return Drama3Reference(codes=codes, text=text)

    def generate_frames(
        self,
        prompt: np.ndarray,
        *,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> tuple[np.ndarray, FinishReason]:
        """Sample frames after the prompt rows: codes [num_codebooks, frames].

        The audio decoder cache of prompt row p is filled with frame p + 1 as its
        label; the last prompt row has none yet, so the first decode step writes it.
        """
        backbone, decoder = self.model.backbone, self.model.audio_decoder
        rows = mx.array(prompt)
        prompt_rows = len(prompt)
        capacity = prompt_rows + max_new_tokens
        backbone_cache = backbone.make_cache(capacity=capacity)
        decoder_cache = decoder.make_cache(capacity=capacity)

        for start in range(0, prompt_rows, PREFILL_CHUNK_ROWS):
            end = min(start + PREFILL_CHUNK_ROWS, prompt_rows)
            chunk = rows[start:end]
            embeddings = decoder.embed_backbone(
                chunk, backbone.embed_tokens(chunk[:, 0])
            )
            hidden = backbone(embeddings[None], backbone_cache)[0]
            paired_end = min(end, prompt_rows - 1)
            if paired_end > start:
                decoder.prefill(
                    hidden[: paired_end - start],
                    chunk[: paired_end - start],
                    rows[start + 1 : paired_end + 1],
                    decoder_cache,
                )
            mx.eval(hidden, _cache_arrays(backbone_cache), _cache_arrays(decoder_cache))

        hidden = hidden[-1:]
        frame = rows[-1:]
        history = rows[-RAS_WINDOW_FRAMES:, 0]
        candidates = decoder.main_candidates
        frames: list[mx.array] = []
        finish_reason: FinishReason = "length"
        for step in range(max_new_tokens):
            sampled = decoder.decode_step(
                backbone.logits(hidden, candidates),
                hidden,
                frame,
                history,
                decoder_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            if sampled[0, 0].item() == self.config.eos_token_id:
                finish_reason = "eos"
                break
            frames.append(sampled)
            frame = sampled
            history = mx.concatenate([history[1:], sampled[:, 0]])
            if step + 1 < max_new_tokens:
                embeddings = decoder.embed_backbone(
                    sampled, backbone.embed_tokens(sampled[:, 0])
                )
                hidden = backbone(embeddings[None], backbone_cache)[0]

        num_codebooks = self.config.audio_decoder.num_codebooks
        if not frames:
            return np.zeros((num_codebooks, 0), dtype=np.int32), finish_reason
        codes = np.array(mx.concatenate(frames)[:, 1:], dtype=np.int32).T
        return np.ascontiguousarray(codes), finish_reason

    def synthesize(
        self,
        text: str,
        references: Sequence[Drama3Reference] = (),
        *,
        duration_seconds: float | None = None,
        max_new_tokens: int | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        single_chunk: bool = False,
    ) -> Iterator[Drama3Chunk]:
        """Generate text chunk by chunk, each chunk's audio as soon as it is done.

        The text is split into chunks of about 300 bytes with at most three
        speakers; the first chunk and the latest one go back into the next prompt
        as history. single_chunk generates the whole text in one pass.

        duration_seconds asks for audio of that length and generates the text in
        one pass; it also keeps takes from stalling in silence after a sentence.
        max_new_tokens caps the frames of each chunk: 1024 by default, or 2.5 times
        the requested duration.
        """
        if duration_seconds is not None:
            if duration_seconds <= 0:
                raise ValueError(
                    f"duration_seconds must be positive; got {duration_seconds}"
                )
            frames = max(1, round(duration_seconds * self.frame_rate))
            text = with_duration_hint(text, frames)
            single_chunk = True
            max_new_tokens = max_new_tokens or math.ceil(frames * RUNAWAY_FACTOR)
        max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS
        # The sampler keeps the best token whatever top_k is, so top_k < 1 would
        # silently turn into greedy decoding.
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1; got {top_k}")
        if max_new_tokens < 1:
            raise ValueError(f"max_new_tokens must be at least 1; got {max_new_tokens}")
        texts = [reference.text for reference in references]
        codes = [reference.codes for reference in references]
        chunks = [text] if single_chunk else split_text(text) or [text]
        # Reject bad text in any chunk before generating: failing at a later chunk
        # would waste the earlier ones.
        for chunk in chunks:
            build_prompt(chunk, self.tokenizer, self.config, texts, codes)

        history = ChunkHistory()
        for chunk in chunks:
            prompt = self._chunk_prompt(chunk, texts, codes, history)
            frames, finish_reason = self.generate_frames(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            mx.clear_cache()
            if frames.shape[1] == 0:
                continue  # the model stopped before any audio frame
            history.push(chunk, frames)
            waveform = self.codec.decode(mx.array(frames)).reshape(-1)
            yield Drama3Chunk(chunk, frames, waveform, finish_reason)

    def _chunk_prompt(
        self,
        text: str,
        reference_texts: list[str],
        reference_codes: list[np.ndarray],
        history: ChunkHistory,
    ) -> np.ndarray:
        """The prompt for one chunk with as much history as fits."""
        error: PromptTooLongError | None = None
        for turns in history.candidates():
            try:
                return build_prompt(
                    text,
                    self.tokenizer,
                    self.config,
                    reference_texts,
                    reference_codes,
                    history=turns,
                )
            except PromptTooLongError as too_long:
                error = too_long
        raise error
