"""Fish Audio Drama 3 inference: text and voice references to a 44.1 kHz waveform.

Ported from upstream fish_omni.generate (one request, one chunk at a time) and the
chunked long-form loop of fish-speech's generate_long_drama3.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
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
from ..models.fish_drama_3.sampling import REPEAT_WINDOW_FRAMES
from ..models.fish_s2_pro.codec import FishS2Codec

# Fish Audio's recommendation under reference sampling: the model's own distribution,
# trimmed only by the epsilon cutoff. top_k 0 removes the cap.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0
# Frames per chunk, about 47 s of audio.
DEFAULT_MAX_NEW_TOKENS = 1024
# A take may run to about 2.5 times its expected duration (technical report, 3.3).
RUNAWAY_FACTOR = 2.5
PREFILL_CHUNK_ROWS = 128
# Lines decoded together by synthesize_batch. Each decode step reads the experts
# its rows route to; by 48 to 64 rows that is nearly all of them, so a step costs
# about what it would with fewer rows. Each line adds about 75 MB of state.
DEFAULT_BATCH_SIZE = 64
# Finished rows leave a decoding batch once they are this fraction of it.
_COMPACT_FINISHED_FRACTION = 0.25

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


@dataclass(frozen=True)
class _PromptHead:
    """Model state after the rows every prompt of one voice setup shares.

    ``rows`` is the head plus the next row, the opening of the current user
    turn: the audio decoder labels each row with the row after it, so the last
    head row needs that one to match too.
    """

    rows: np.ndarray
    backbone: list[BoundedKVCache | DeltaNetState]
    decoder: list[BoundedKVCache]

    def matches(self, prompt: np.ndarray) -> bool:
        return len(prompt) > len(self.rows) - 1 and np.array_equal(
            prompt[: len(self.rows)], self.rows
        )


def _head_length(prompt: np.ndarray, turn_start_id: int) -> int:
    """Rows before the current user turn: the system turn, references and
    history. The prompt ends with that turn and the assistant opening, each
    starting with <|im_start|>; 0 when there is nothing before them."""
    starts = np.flatnonzero(prompt[:, 0] == turn_start_id)
    return int(starts[-2]) if len(starts) >= 2 else 0


def _copy_layer(
    state: BoundedKVCache | DeltaNetState, capacity: int
) -> BoundedKVCache | DeltaNetState:
    """An independent copy that can grow to capacity rows. DeltaNet states are
    replaced, never written in place, so sharing their arrays is safe."""
    if isinstance(state, DeltaNetState):
        return DeltaNetState(conv=state.conv, recurrent=state.recurrent)
    keys, values = state.fetch()
    return BoundedKVCache.from_values(keys, values, capacity=capacity)


def _batched_caches(
    caches: Sequence[list[BoundedKVCache | DeltaNetState | None]],
    lengths: Sequence[int],
    capacity: int,
) -> list[BoundedKVCache | DeltaNetState]:
    """One batch-wide cache per layer from single-sequence caches, which it
    consumes: each layer's sources are dropped as soon as it is built, so the two
    copies never coexist beyond one layer.

    Recurrent state stacks as is. Attention caches are right-aligned: sequence i,
    of lengths[i] rows, fills the slots up to the longest, after pads of
    longest - lengths[i] empty slots that attention masks.
    """
    longest = max(lengths)
    batched = []
    for index in range(len(caches[0])):
        layer = [sequence[index] for sequence in caches]
        for sequence in caches:
            sequence[index] = None
        if isinstance(layer[0], DeltaNetState):
            batched.append(
                DeltaNetState(
                    conv=mx.concatenate([state.conv for state in layer]),
                    recurrent=mx.concatenate([state.recurrent for state in layer]),
                )
            )
            mx.eval(batched[-1].conv, batched[-1].recurrent)
            continue
        keys, values = [], []
        for state, length in zip(layer, lengths, strict=True):
            stored_keys, stored_values = state.fetch()
            padding = [(0, 0), (longest - length, capacity - longest), (0, 0), (0, 0)]
            keys.append(mx.pad(stored_keys[:, :length], padding))
            values.append(mx.pad(stored_values[:, :length], padding))
        batched.append(
            BoundedKVCache(
                keys=mx.concatenate(keys),
                values=mx.concatenate(values),
                offset=longest,
            )
        )
        mx.eval(batched[-1].keys, batched[-1].values)
    return batched


def _select_rows(cache: BoundedKVCache | DeltaNetState, rows: mx.array) -> None:
    if isinstance(cache, DeltaNetState):
        cache.conv, cache.recurrent = cache.conv[rows], cache.recurrent[rows]
    else:
        cache.keys, cache.values = cache.keys[rows], cache.values[rows]


@mx.compile
def _split_keys(keys: mx.array, count: int) -> mx.array:
    """Each row's key split into count keys: [rows, count]."""
    return mx.vmap(lambda key: mx.random.split(key, num=count))(keys)


def _seed_key(seed: int) -> mx.array:
    if not 0 <= seed < 2**32:
        raise ValueError(f"a seed must be an unsigned 32-bit integer; got {seed}")
    return mx.random.key(seed)


def _per_line(value, count: int, name: str) -> list:
    """value as one entry per line: a sequence of count, or one value for all."""
    if isinstance(value, (list, tuple)):
        if len(value) != count:
            raise ValueError(f"{name} needs one entry per line; got {len(value)} for {count}")
        return list(value)
    return [value] * count


def _check_sampling(top_k: int, max_new_tokens: int) -> None:
    if top_k < 0:
        raise ValueError(f"top_k must be at least 0 (0: no cap); got {top_k}")
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be at least 1; got {max_new_tokens}")


def _random_key() -> mx.array:
    """A key drawn from the global generator, which mx.random.seed controls."""
    return mx.random.key(int(mx.random.randint(0, 2**31 - 1, ()).item()))


def _cache_arrays(caches: Sequence[BoundedKVCache | DeltaNetState]) -> list[mx.array]:
    arrays = []
    for cache in caches:
        if isinstance(cache, DeltaNetState):
            arrays += [cache.conv, cache.recurrent]
        else:
            arrays += [cache.keys, cache.values]
    return arrays


@dataclass
class _Prefilled:
    """One prompt run through the model, ready for its first decode step."""

    rows: int
    hidden: mx.array  # [1, hidden]: the last prompt row's
    frame: mx.array  # [1, 1 + num_codebooks]: the last prompt row
    history: mx.array  # [REPEAT_WINDOW_FRAMES]: the latest main tokens
    backbone: list[BoundedKVCache | DeltaNetState]
    decoder: list[BoundedKVCache]


@dataclass
class FishDrama3Runtime:
    model: FishDrama3Model
    tokenizer: Tokenizer
    codec: FishS2Codec
    # The shared head of the last prompt, reused while prompts keep it.
    _head: _PromptHead | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_dir(cls, model_dir: str | Path, *, dense_int8: bool = False) -> FishDrama3Runtime:
        """Load the upstream release as is.

        dense_int8 packs the dense BF16 projections as int8 once loaded, for faster
        decoding; see load_fish_drama_3_model."""
        resolved = Path(model_dir)
        return cls(
            model=load_fish_drama_3_model(resolved, dense_int8=dense_int8),
            tokenizer=Tokenizer.from_file(str(resolved / "tokenizer.json")),
            # The release ships the S2 Pro codec, byte for byte, as codec.pth.
            codec=FishS2Codec.from_codec_pth(resolved / "codec.pth"),
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
        key: mx.array | None = None,
    ) -> tuple[np.ndarray, FinishReason]:
        """Sample frames after the prompt rows: codes [num_codebooks, frames].

        key seeds the draws; None takes one from the global generator. Everything
        before the current user turn is prefilled apart and kept: a later prompt
        that starts with the same rows reuses it, with the same result.
        """
        key = _random_key() if key is None else key
        return self._decode(
            [self._prefill_prompt(prompt)],
            [max_new_tokens],
            key[None],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )[0]

    def _prefill_prompt(self, prompt: np.ndarray) -> _Prefilled:
        """Run one prompt through the model, reusing the kept head when it matches.

        The audio decoder cache of prompt row p is filled with frame p + 1 as its
        label; the last prompt row has none yet, so the first decode step writes it.
        """
        backbone, decoder = self.model.backbone, self.model.audio_decoder
        rows = mx.array(prompt)
        prompt_rows = len(prompt)
        head_rows = _head_length(prompt, self._turn_start_id)
        head = self._head
        if head_rows and head is not None and len(head.rows) == head_rows + 1 and head.matches(prompt):
            backbone_cache = [_copy_layer(state, prompt_rows) for state in head.backbone]
            decoder_cache = [_copy_layer(state, prompt_rows) for state in head.decoder]
        else:
            # The caches hold activations, in the weights' dtype.
            dtype = backbone.embed_tokens.weight.dtype
            backbone_cache = backbone.make_cache(capacity=prompt_rows, dtype=dtype)
            decoder_cache = decoder.make_cache(capacity=prompt_rows, dtype=dtype)
            if head_rows:
                self._prefill(rows, 0, head_rows, backbone_cache, decoder_cache)
                self._head = _PromptHead(
                    rows=np.array(prompt[: head_rows + 1]),
                    backbone=[_copy_layer(state, head_rows) for state in backbone_cache],
                    decoder=[_copy_layer(state, head_rows) for state in decoder_cache],
                )
                mx.eval(_cache_arrays(self._head.backbone), _cache_arrays(self._head.decoder))
        hidden = self._prefill(rows, head_rows, prompt_rows, backbone_cache, decoder_cache)
        return _Prefilled(
            rows=prompt_rows,
            hidden=hidden[-1:],
            frame=rows[-1:],
            # A short prompt's window starts with ids no token has.
            history=mx.pad(
                rows[-REPEAT_WINDOW_FRAMES:, 0],
                [(max(0, REPEAT_WINDOW_FRAMES - prompt_rows), 0)],
                constant_values=-1,
            ),
            backbone=backbone_cache,
            decoder=decoder_cache,
        )

    def _decode(
        self,
        requests: Sequence[_Prefilled],
        budgets: Sequence[int],
        keys: mx.array,
        *,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> list[tuple[np.ndarray, FinishReason]]:
        """Decode prefilled prompts together, one frame of each per step.

        Each request stops at its eos or after budgets[i] frames and draws with its
        own key, whatever the requests beside it. Prompts of
        different lengths share right-aligned caches: request i's cached rows start
        at slot pads[i]. Finished rows leave the batch once enough of them have.
        The requests' caches are consumed into the batch's.
        """
        backbone, decoder = self.model.backbone, self.model.audio_decoder
        num_codebooks = self.config.audio_decoder.num_codebooks
        count = len(requests)
        lengths = [request.rows for request in requests]
        longest, most = max(lengths), max(budgets)
        pads = [longest - length for length in lengths]
        backbone_cache = _batched_caches(
            [request.backbone for request in requests], lengths, longest + most
        )
        decoder_cache = _batched_caches(
            [request.decoder for request in requests],
            [length - 1 for length in lengths],
            longest - 1 + most,
        )
        hidden = mx.concatenate([request.hidden for request in requests])
        frame = mx.concatenate([request.frame for request in requests])
        history = mx.stack([request.history for request in requests])
        candidates = decoder.main_candidates
        eos = self.config.eos_token_id
        generated = mx.zeros((count, most, 1 + num_codebooks), dtype=mx.int32)
        produced = [0] * count
        reasons: list[FinishReason] = ["length"] * count
        active = list(range(count))
        finished: set[int] = set()
        for step in range(most):
            pad_rows = mx.array(pads)
            draws = _split_keys(keys, num_codebooks + 1)
            keys = draws[:, 0]
            sampled = decoder.decode_step(
                backbone.logits(hidden, candidates),
                hidden,
                frame,
                history,
                decoder_cache,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                # One unpadded sequence runs as the single-sequence tower does.
                pads=pad_rows if len(active) > 1 or any(pads) else None,
                keys=draws[:, 1:],
            )
            generated[mx.array(active), step] = sampled
            keep = []
            # One host read per step, for each request's own eos.
            for row, main in enumerate(sampled[:, 0].tolist()):
                request = active[row]
                if request in finished:
                    continue
                if main == eos:
                    reasons[request] = "eos"
                    finished.add(request)
                    continue
                produced[request] += 1
                if produced[request] == budgets[request]:
                    finished.add(request)
                else:
                    keep.append(row)
            if not keep:
                break
            if len(active) - len(keep) >= len(active) * _COMPACT_FINISHED_FRACTION:
                rows = mx.array(keep)
                for cache in (*backbone_cache, *decoder_cache):
                    _select_rows(cache, rows)
                sampled, history, keys = sampled[rows], history[rows], keys[rows]
                active = [active[row] for row in keep]
                pads = [pads[row] for row in keep]
                pad_rows = mx.array(pads)
            frame = sampled
            history = mx.concatenate([history[:, 1:], sampled[:, :1]], axis=1)
            embeddings = decoder.embed_backbone(
                sampled, backbone.embed_tokens(sampled[:, 0])
            )
            hidden = backbone(
                embeddings[:, None],
                backbone_cache,
                pad_rows if any(pads) else None,
            )[:, 0]
            mx.eval(
                generated,
                hidden,
                keys,
                history,
                _cache_arrays(backbone_cache),
                _cache_arrays(decoder_cache),
            )
        results = []
        for request in range(count):
            codes = np.array(generated[request, : produced[request], 1:], dtype=np.int32)
            results.append((np.ascontiguousarray(codes.T), reasons[request]))
        return results

    @property
    def _turn_start_id(self) -> int:
        return self.tokenizer.token_to_id("<|im_start|>")

    def _prefill(
        self,
        rows: mx.array,
        start: int,
        end: int,
        backbone_cache: list[BoundedKVCache | DeltaNetState],
        decoder_cache: list[BoundedKVCache],
    ) -> mx.array:
        """Run prompt rows start..end-1 through the backbone, and label each of them
        but the prompt's last for the audio decoder. Returns the last row's hidden."""
        backbone, decoder = self.model.backbone, self.model.audio_decoder
        last_labelled = len(rows) - 1
        for chunk_start in range(start, end, PREFILL_CHUNK_ROWS):
            chunk_end = min(chunk_start + PREFILL_CHUNK_ROWS, end)
            chunk = rows[chunk_start:chunk_end]
            embeddings = decoder.embed_backbone(chunk, backbone.embed_tokens(chunk[:, 0]))
            hidden = backbone(embeddings[None], backbone_cache)[0]
            paired_end = min(chunk_end, last_labelled)
            if paired_end > chunk_start:
                decoder.prefill(
                    hidden[: paired_end - chunk_start],
                    chunk[: paired_end - chunk_start],
                    rows[chunk_start + 1 : paired_end + 1],
                    decoder_cache,
                )
            mx.eval(hidden, _cache_arrays(backbone_cache), _cache_arrays(decoder_cache))
        return hidden

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
        seed: int | None = None,
    ) -> Iterator[Drama3Chunk]:
        """Generate text chunk by chunk, each chunk's audio as soon as it is done.

        The text is split into chunks of about 300 bytes with at most three
        speakers; the first chunk and the latest one go back into the next prompt
        as history. single_chunk generates the whole text in one pass.

        duration_seconds asks for audio of that length and generates the text in
        one pass; it also keeps takes from stalling in silence after a sentence.
        max_new_tokens caps the frames of each chunk: 1024 by default, or 2.5 times
        the requested duration. seed fixes the draws of a line, as in
        synthesize_batch; None takes them from the global generator.
        """
        if duration_seconds is not None:
            text, max_new_tokens = self._timed(text, duration_seconds, max_new_tokens)
            single_chunk = True
        max_new_tokens = max_new_tokens or DEFAULT_MAX_NEW_TOKENS
        _check_sampling(top_k, max_new_tokens)
        key = _random_key() if seed is None else _seed_key(seed)
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
            # The first chunk draws with the line's key, as a batched line does.
            chunk_key, key = key, mx.random.split(key, 1)[0]
            frames, finish_reason = self.generate_frames(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                key=chunk_key,
            )
            mx.clear_cache()
            if frames.shape[1] == 0:
                continue  # the model stopped before any audio frame
            history.push(chunk, frames)
            waveform = self.codec.decode(mx.array(frames)).reshape(-1)
            yield Drama3Chunk(chunk, frames, waveform, finish_reason)

    def synthesize_batch(
        self,
        texts: Sequence[str],
        references: Sequence[Drama3Reference] = (),
        *,
        seeds: Sequence[int],
        duration_seconds: float | Sequence[float | None] | None = None,
        max_new_tokens: int | Sequence[int | None] | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        top_k: int = DEFAULT_TOP_K,
        top_p: float = DEFAULT_TOP_P,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[Drama3Chunk]:
        """Generate independent lines together; results keep input order.

        Every line shares the references and is one pass, as with single_chunk.
        duration_seconds and max_new_tokens are one value for every line or one per
        line. Lines decode batch_size at a time, grouped by frame budget so short
        lines do not wait for long ones. A line draws with its own seed, whatever
        the lines beside it; only rounding in batched bf16 matmuls can make a draw
        differ from the line generated alone.
        """
        if isinstance(texts, str):
            raise TypeError("texts must be a sequence of strings, not one string")
        count = len(texts)
        durations = _per_line(duration_seconds, count, "duration_seconds")
        caps = _per_line(max_new_tokens, count, "max_new_tokens")
        if len(seeds) != count:
            raise ValueError(f"seeds needs one entry per line; got {len(seeds)} for {count}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1; got {batch_size}")
        reference_texts = [reference.text for reference in references]
        reference_codes = [reference.codes for reference in references]
        prompts, budgets = [], []
        # Every line is checked before any is generated.
        for index, (text, duration, cap) in enumerate(zip(texts, durations, caps)):
            if duration is not None:
                text, cap = self._timed(text, duration, cap)
            budget = cap or DEFAULT_MAX_NEW_TOKENS
            _check_sampling(top_k, budget)
            try:
                prompts.append(
                    build_prompt(
                        text, self.tokenizer, self.config, reference_texts, reference_codes
                    )
                )
            except ValueError as error:
                raise ValueError(f"line {index}: {error}") from error
            budgets.append(budget)
        keys = [_seed_key(seed) for seed in seeds]

        lines: list[Drama3Chunk | None] = [None] * count
        order = sorted(range(count), key=budgets.__getitem__)
        for start in range(0, count, batch_size):
            group = order[start : start + batch_size]
            decoded = self._decode(
                [self._prefill_prompt(prompts[i]) for i in group],
                [budgets[i] for i in group],
                mx.stack([keys[i] for i in group]),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            mx.clear_cache()
            for i, (codes, finish_reason) in zip(group, decoded, strict=True):
                waveform = (
                    self.codec.decode(mx.array(codes)).reshape(-1)
                    if codes.shape[1]
                    else mx.zeros((0,), dtype=mx.float32)
                )
                lines[i] = Drama3Chunk(texts[i], codes, waveform, finish_reason)
        return lines

    def _timed(
        self, text: str, duration_seconds: float, max_new_tokens: int | None
    ) -> tuple[str, int]:
        """text with its duration hint, and its frame budget: max_new_tokens or 2.5
        times the requested frames."""
        if duration_seconds <= 0:
            raise ValueError(f"duration_seconds must be positive; got {duration_seconds}")
        frames = max(1, round(duration_seconds * self.frame_rate))
        return (
            with_duration_hint(text, frames),
            max_new_tokens or math.ceil(frames * RUNAWAY_FACTOR),
        )

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
