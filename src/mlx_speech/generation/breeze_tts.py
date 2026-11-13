"""Breeze TTS 2 generation.

Offline and streaming output share one code-frame loop. Streaming decodes
those frames through codec state instead of waiting for the full utterance.

Voice design is ``[speaker]text``. An instruction uses the official
``<ins_bos>...<ins_eos>`` span. Single-branch CFG is
``uncond + scale * (cond - uncond)`` on the backbone and the depth decoder.
Scale 1 keeps the conditional branch only.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from mlx_speech.models._cache import BoundedKVCache
from mlx_speech.models.breeze_tts.backbone import BreezeBackbone, EmbeddingTable
from mlx_speech.models.breeze_tts.codec import (
    CodecDecodeStream,
    SpeechCodec,
    codec_parameter_key,
)
from mlx_speech.models.breeze_tts.checkpoint import is_quantizable, read_quantization
from mlx_speech.models.breeze_tts.config import BreezeRuntimeConfig, load_runtime_config
from mlx_speech.models.breeze_tts.depth import BreezeDepthDecoder
from mlx_speech.models.breeze_tts.text_encoder import BreezeTextEncoder

CODEC_CODEBOOK_SIZE = 2048
SAMPLE_RATE = 24_000
FRAME_RATE = 12.5


@dataclass(frozen=True)
class BreezeAudio:
    samples: mx.array
    sample_rate: int
    frames: int


class BreezeSpeech(nn.Module):
    def __init__(self, config: BreezeRuntimeConfig) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = BreezeTextEncoder(config.text_encoder)
        self.text_encoder_proj = nn.Linear(
            config.text_encoder.hidden_size, config.backbone.hidden_size, bias=False
        )
        self.backbone_model = BreezeBackbone(config)
        self.depth_decoder = BreezeDepthDecoder(config)
        self.embed_text_tokens = EmbeddingTable(
            config.text_vocab_size, config.backbone.hidden_size
        )
        self.lm_head = nn.Linear(
            config.backbone.hidden_size, config.vocab_size + 1, bias=False
        )


def load_breeze_speech(model_dir: Path) -> tuple[BreezeSpeech, SpeechCodec, Tokenizer]:
    model_dir = model_dir.expanduser().resolve()
    config = load_runtime_config(model_dir)
    speech = BreezeSpeech(config)
    quantization = read_quantization(model_dir)
    if quantization is not None:
        eligible = tuple(
            sorted(
                path
                for path, module in speech.named_modules()
                if isinstance(module, nn.Linear) and is_quantizable(path)
            )
        )
        if quantization.paths != eligible:
            raise ValueError("Breeze int8 metadata does not match the model's projections.")
        selected = set(quantization.paths)
        nn.quantize(
            speech,
            group_size=quantization.group_size,
            bits=quantization.bits,
            mode=quantization.mode,
            class_predicate=lambda path, module: path in selected,
        )
    main = mx.load(str(model_dir / "model.safetensors"))
    speech.load_weights(list(main.items()), strict=True)
    codec = SpeechCodec()
    codec_weights = mx.load(str(model_dir / "audio_tokenizer" / "model.safetensors"))
    codec_parameters = [
        (codec_parameter_key(key), value) for key, value in codec_weights.items()
    ]
    codec.load_weights(codec_parameters, strict=True)
    mx.eval(tree_flatten(speech.parameters()), tree_flatten(codec.parameters()))
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    return speech, codec, tokenizer


def _iter_code_frames(
    speech: BreezeSpeech,
    codec: SpeechCodec,
    tokenizer: Tokenizer,
    text: str,
    *,
    speaker: str = "S0",
    instruction: str | None = None,
    cfg_scale: float = 1.0,
    max_frames: int = 80,
    temperature: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    seed: int = 0,
    ref_audio: mx.array | None = None,
    ref_text: str | None = None,
) -> Iterator[mx.array]:
    """Yield one ``(num_codebooks,)`` frame at a time, stopping before EOS.

    Running out of ``max_frames`` before EOS warns: the speech may be cut off.
    """

    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if not _finite_positive(cfg_scale):
        raise ValueError("cfg_scale must be finite and greater than 0.")
    if not _finite_positive(temperature):
        raise ValueError("temperature must be finite and greater than 0.")
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer; 0 disables it.")
    if not _finite_positive(repetition_penalty):
        raise ValueError("repetition_penalty must be finite and greater than 0.")
    direction = (instruction or "").strip()
    if cfg_scale != 1.0 and not direction:
        raise ValueError("cfg_scale other than 1 requires an instruction.")
    reference = _validated_reference(ref_audio, ref_text)
    mx.random.seed(seed)
    guided = cfg_scale != 1.0
    prompt = _prompt_embeddings(
        speech,
        codec,
        tokenizer,
        text,
        speaker=speaker,
        instruction=direction or None,
        reference=reference,
    )
    cache = speech.backbone_model.make_cache(batch_size=1, dtype=prompt.dtype)
    hidden = speech.backbone_model(prompt, cache)[:, -1, :]
    uncond_cache = None
    uncond_hidden = None
    if guided:
        plain = _prompt_embeddings(
            speech,
            codec,
            tokenizer,
            text,
            speaker=speaker,
            instruction=None,
            reference=reference,
        )
        uncond_cache = speech.backbone_model.make_cache(batch_size=1, dtype=plain.dtype)
        uncond_hidden = speech.backbone_model(plain, uncond_cache)[:, -1, :]
    _realize([hidden, uncond_hidden], [cache, uncond_cache])
    positions = pad_lengths = None
    if uncond_cache is not None:
        # The branches decode as one batch of two from here on.
        cache, positions, pad_lengths = _join_branches(cache, uncond_cache)
        uncond_cache = None
    first_codes: list[int] = []
    for _ in range(max_frames):
        logits = _backbone_scores(
            speech.lm_head(hidden),
            None if uncond_hidden is None else speech.lm_head(uncond_hidden),
            cfg_scale,
            first_codes,
            repetition_penalty,
        )
        first = _sample(logits, temperature=temperature, top_k=top_k)
        if int(first.item()) == speech.config.backbone_eos_id:
            break
        frame = _depth_frame(
            speech,
            hidden,
            first,
            temperature=temperature,
            top_k=top_k,
            uncond_hidden=uncond_hidden,
            cfg_scale=cfg_scale,
        )
        first_codes.append(int(first.item()))
        yield frame
        audio = speech.backbone_model.embed_audio(frame.reshape(1, 1, -1))
        if positions is None:
            hidden = speech.backbone_model(audio, cache)[:, -1, :]
        else:
            both = speech.backbone_model(
                mx.broadcast_to(audio, (2, *audio.shape[1:])),
                cache,
                position_ids=positions[:, None],
                pad_lengths=pad_lengths,
            )[:, -1, :]
            hidden, uncond_hidden = both[:1], both[1:]
            positions = positions + 1
        _realize([frame, hidden, uncond_hidden], [cache])
    else:
        warnings.warn(
            f"Breeze reached the {max_frames}-frame budget "
            f"({max_frames / FRAME_RATE:.1f} s) before EOS; the speech may be "
            "cut off. Raise max_new_tokens for longer text.",
            RuntimeWarning,
            stacklevel=2,
        )


def generate_breeze(
    speech: BreezeSpeech,
    codec: SpeechCodec,
    tokenizer: Tokenizer,
    text: str,
    *,
    speaker: str = "S0",
    instruction: str | None = None,
    cfg_scale: float = 1.0,
    max_frames: int = 80,
    temperature: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    seed: int = 0,
    ref_audio: mx.array | None = None,
    ref_text: str | None = None,
) -> BreezeAudio:
    """Generate one utterance. ``cfg_scale`` other than 1 requires an instruction."""

    frames = list(
        _iter_code_frames(
            speech,
            codec,
            tokenizer,
            text,
            speaker=speaker,
            instruction=instruction,
            cfg_scale=cfg_scale,
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            ref_audio=ref_audio,
            ref_text=ref_text,
        )
    )
    if not frames:
        return BreezeAudio(mx.zeros((0,), dtype=mx.float32), SAMPLE_RATE, 0)
    codes = mx.stack(frames, axis=0)[None, :, :]
    audio = codec.decode(mx.transpose(codes, (0, 2, 1)))
    waveform = audio[0, 0].astype(mx.float32)
    mx.eval(waveform)
    return BreezeAudio(waveform, SAMPLE_RATE, len(frames))


def generate_breeze_stream(
    speech: BreezeSpeech,
    codec: SpeechCodec,
    tokenizer: Tokenizer,
    text: str,
    *,
    speaker: str = "S0",
    instruction: str | None = None,
    cfg_scale: float = 1.0,
    max_frames: int = 80,
    temperature: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    seed: int = 0,
    ref_audio: mx.array | None = None,
    ref_text: str | None = None,
    chunk_frames: int = 4,
) -> Iterator[mx.array]:
    """Yield mono PCM chunks. Each chunk is decoded before later frames exist."""

    if (
        isinstance(chunk_frames, bool)
        or not isinstance(chunk_frames, int)
        or chunk_frames <= 0
    ):
        raise ValueError("chunk_frames must be a positive integer.")
    stream = CodecDecodeStream(codec.decoder)
    pending: list[mx.array] = []
    try:
        for frame in _iter_code_frames(
            speech,
            codec,
            tokenizer,
            text,
            speaker=speaker,
            instruction=instruction,
            cfg_scale=cfg_scale,
            max_frames=max_frames,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            seed=seed,
            ref_audio=ref_audio,
            ref_text=ref_text,
        ):
            pending.append(frame)
            if len(pending) == chunk_frames:
                yield _emit_stream_chunk(stream, pending)
                pending = []
        if pending:
            yield _emit_stream_chunk(stream, pending)
    finally:
        stream.close()


def _emit_stream_chunk(stream: CodecDecodeStream, frames: list[mx.array]) -> mx.array:
    codes = mx.transpose(mx.stack(frames, axis=0), (1, 0))[None, :, :]
    waveform = stream.push(codes)[0, 0].astype(mx.float32)
    mx.eval(waveform)
    return waveform


def _prompt_text(speaker: str, text: str, instruction: str | None) -> str:
    prefix = (
        speaker if speaker.startswith("[") and speaker.endswith("]") else f"[{speaker}]"
    )
    if instruction:
        return f"{prefix}<ins_bos>{instruction}<ins_eos>{text}"
    return f"{prefix}{text}"


def _finite_positive(value: float) -> bool:
    return value > 0 and value == value and value not in {float("inf"), float("-inf")}


def _validated_reference(
    ref_audio: mx.array | None, ref_text: str | None
) -> tuple[mx.array, str] | None:
    transcript = (ref_text or "").strip()
    has_audio = ref_audio is not None
    if has_audio != bool(transcript):
        raise ValueError("ref_audio and ref_text must be provided together.")
    if ref_audio is None:
        return None
    return ref_audio, transcript


def _prompt_embeddings(
    speech: BreezeSpeech,
    codec: SpeechCodec,
    tokenizer: Tokenizer,
    text: str,
    *,
    speaker: str,
    instruction: str | None,
    reference: tuple[mx.array, str] | None,
) -> mx.array:
    if reference is None:
        return _text_embeddings(
            speech, tokenizer, _prompt_text(speaker, text, instruction)
        )
    ref_audio, ref_text = reference
    codes = codec.encode(ref_audio)
    left = _segment_embeddings(speech, tokenizer, _prompt_text(speaker, ref_text, None))
    right = _segment_embeddings(
        speech, tokenizer, _prompt_text(speaker, text, instruction)
    )
    audio = speech.backbone_model.embed_audio(codes[None, :, :])[0]
    eos = speech.backbone_model.embed_audio(
        mx.zeros((1, 1, codes.shape[1]), dtype=mx.int32)
    )[0]
    return mx.concatenate([left, audio, eos, right], axis=0)[None, :, :]


def _text_embeddings(speech: BreezeSpeech, tokenizer: Tokenizer, text: str) -> mx.array:
    token_ids = tokenizer.encode(text, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id is not None and (not token_ids or token_ids[0] != bos_id):
        token_ids = [bos_id, *token_ids]
    hidden = speech.text_encoder(mx.array(token_ids, dtype=mx.int32)[None, :])
    return speech.text_encoder_proj(hidden)


def _segment_embeddings(
    speech: BreezeSpeech, tokenizer: Tokenizer, text: str
) -> mx.array:
    rendered = tokenizer.decode(
        tokenizer.encode(text, add_special_tokens=True).ids, skip_special_tokens=False
    )
    token_ids = tokenizer.encode(rendered, add_special_tokens=False).ids
    hidden = speech.text_encoder(mx.array(token_ids, dtype=mx.int32)[None, :])
    return speech.text_encoder_proj(hidden)[0]


def _join_branches(
    conditional: list[BoundedKVCache], unconditional: list[BoundedKVCache]
) -> tuple[list[BoundedKVCache], mx.array, mx.array]:
    """One batch-of-two cache from two prefilled branches: row 0 conditional,
    row 1 unconditional. The shorter prompt is left-padded, and the returned
    pad lengths mask those slots. Each row keeps its own next position, so
    rotary phases match the unbatched branches."""

    lengths = (conditional[0].offset, unconditional[0].offset)
    longest = max(lengths)
    joined = []
    for cond, uncond in zip(conditional, unconditional, strict=True):
        rows = []
        for layer, length in ((cond, lengths[0]), (uncond, lengths[1])):
            pad = [(0, 0), (longest - length, 0), (0, 0), (0, 0)]
            rows.append(
                (
                    mx.pad(layer.keys[:, :length], pad),
                    mx.pad(layer.values[:, :length], pad),
                )
            )
        joined.append(
            BoundedKVCache.from_values(
                mx.concatenate([rows[0][0], rows[1][0]], axis=0),
                mx.concatenate([rows[0][1], rows[1][1]], axis=0),
                capacity=cond.capacity,
                max_capacity=cond.max_capacity,
                growth_step=cond.growth_step,
            )
        )
    positions = mx.array(lengths, dtype=mx.int32)
    return joined, positions, longest - positions


def _depth_frame(
    speech: BreezeSpeech,
    backbone_hidden: mx.array,
    first: mx.array,
    *,
    temperature: float,
    top_k: int,
    uncond_hidden: mx.array | None = None,
    cfg_scale: float = 1.0,
) -> mx.array:
    """One frame of residual codebooks.

    Under guidance both branches see the same sampled codes at the same
    positions and differ only in their backbone state, so they run as one
    batch of two: row 0 conditional, row 1 unconditional.
    """

    depth = speech.depth_decoder
    hidden = (
        backbone_hidden
        if uncond_hidden is None
        else mx.concatenate([backbone_hidden, uncond_hidden], axis=0)
    )
    rows = int(hidden.shape[0])
    cache = depth.make_cache(batch_size=rows, dtype=hidden.dtype)

    def embed(code: mx.array, codebook: int) -> mx.array:
        embedding = depth.embed_code(code, codebook)
        return mx.broadcast_to(embedding, (rows, embedding.shape[-1]))

    def scores(logits: mx.array) -> mx.array:
        return _depth_scores(logits[:1], logits[1:] if rows > 1 else None, cfg_scale)

    codes = [first]
    logits = scores(depth.begin_frame(hidden, embed(first, 0), cache))
    for codebook in range(1, speech.config.num_codebooks):
        sampled = _sample(logits, temperature=temperature, top_k=top_k)
        codes.append(sampled)
        if codebook == speech.config.num_codebooks - 1:
            break
        logits = scores(
            depth.step(embed(sampled, codebook), codebook_index=codebook, cache=cache)
        )
    return mx.concatenate([code.reshape((1,)) for code in codes], axis=0)


def _backbone_scores(
    conditional: mx.array,
    unconditional: mx.array | None,
    cfg_scale: float,
    tokens: list[int],
    penalty: float,
) -> mx.array:
    """Combine branches, then penalize, then mask reserved ids.

    Repetition penalty is sign-dependent, so applying it to each branch and
    combining afterwards is not the official CFG distribution.
    """

    guided = _guide(conditional, unconditional, cfg_scale)
    return _mask_reserved(_penalize(guided, tokens, penalty), keep_eos=True)


def _depth_scores(
    conditional: mx.array, unconditional: mx.array | None, cfg_scale: float
) -> mx.array:
    return _mask_reserved(_guide(conditional, unconditional, cfg_scale), keep_eos=False)


def _guide(
    conditional: mx.array, unconditional: mx.array | None, cfg_scale: float
) -> mx.array:
    conditional = conditional.astype(mx.float32)
    if unconditional is None or cfg_scale == 1.0:
        return conditional
    unconditional = unconditional.astype(mx.float32)
    return unconditional + cfg_scale * (conditional - unconditional)


def _realize(arrays: list[mx.array | None], caches: list) -> None:
    """Materialize one frame so later frames do not keep its graph alive."""

    pending = [array for array in arrays if array is not None]
    for cache in caches:
        if not cache:
            continue
        for layer in cache:
            pending.append(layer.keys)
            pending.append(layer.values)
    if pending:
        mx.eval(*pending)


def _mask_reserved(logits: mx.array, *, keep_eos: bool) -> mx.array:
    masked = logits.astype(mx.float32)
    masked[..., CODEC_CODEBOOK_SIZE:2051] = -1.0e9
    if not keep_eos and masked.shape[-1] > 2051:
        masked = masked[..., :2051]
    return masked


def _penalize(logits: mx.array, tokens: list[int], penalty: float) -> mx.array:
    if penalty == 1.0 or not tokens:
        return logits
    ids = mx.array(
        sorted({token for token in tokens if 0 <= token < logits.shape[-1]}),
        dtype=mx.int32,
    )
    selected = logits[..., ids]
    adjusted = mx.where(selected < 0, selected * penalty, selected / penalty)
    return logits.at[..., ids].add(adjusted - selected)


def _sample(logits: mx.array, *, temperature: float, top_k: int) -> mx.array:
    values = logits / temperature
    if top_k > 0 and top_k < values.shape[-1]:
        # mx.topk does not sort its result; the threshold is its minimum.
        cutoff = mx.min(mx.topk(values, top_k), axis=-1, keepdims=True)
        values = mx.where(values < cutoff, mx.array(-1.0e9, dtype=values.dtype), values)
    token = mx.random.categorical(nn.log_softmax(values, axis=-1))
    return token.reshape((1,))
