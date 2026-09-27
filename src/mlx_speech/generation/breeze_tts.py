"""Offline Breeze TTS 2 generation.

Voice design is ``[speaker]text``. An instruction uses the official
``<ins_bos>...<ins_eos>`` span. Single-branch CFG is
``uncond + scale * (cond - uncond)`` on the backbone and the depth decoder.
Scale 1 keeps the conditional branch only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from mlx_speech.models.breeze_tts.backbone import BreezeBackbone, EmbeddingTable
from mlx_speech.models.breeze_tts.codec import SpeechCodec, codec_parameter_key
from mlx_speech.models.breeze_tts.config import BreezeRuntimeConfig, load_runtime_config
from mlx_speech.models.breeze_tts.depth import BreezeDepthDecoder
from mlx_speech.models.breeze_tts.text_encoder import BreezeTextEncoder

CODEC_CODEBOOK_SIZE = 2048
SAMPLE_RATE = 24_000


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

    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    if not _finite_positive(cfg_scale):
        raise ValueError("cfg_scale must be finite and greater than 0.")
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
    frames: list[mx.array] = []
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
        frames.append(frame)
        first_codes.append(int(first.item()))
        audio = speech.backbone_model.embed_audio(frame.reshape(1, 1, -1))
        hidden = speech.backbone_model(audio, cache)[:, -1, :]
        if uncond_cache is not None:
            uncond_hidden = speech.backbone_model(audio, uncond_cache)[:, -1, :]
        _realize([frame, hidden, uncond_hidden], [cache, uncond_cache])
    if not frames:
        return BreezeAudio(mx.zeros((0,), dtype=mx.float32), SAMPLE_RATE, 0)
    codes = mx.stack(frames, axis=0)[None, :, :]
    audio = codec.decode(mx.transpose(codes, (0, 2, 1)))
    waveform = audio[0, 0].astype(mx.float32)
    mx.eval(waveform)
    return BreezeAudio(waveform, SAMPLE_RATE, len(frames))


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
    depth = speech.depth_decoder
    cache = depth.make_cache(batch_size=1, dtype=backbone_hidden.dtype)
    uncond_cache = None
    if uncond_hidden is not None:
        uncond_cache = depth.make_cache(batch_size=1, dtype=uncond_hidden.dtype)
    codes = [first]
    logits = _depth_scores(
        depth.begin_frame(backbone_hidden, depth.embed_code(first, 0), cache),
        None
        if uncond_cache is None or uncond_hidden is None
        else depth.begin_frame(uncond_hidden, depth.embed_code(first, 0), uncond_cache),
        cfg_scale,
    )
    for codebook in range(1, speech.config.num_codebooks):
        sampled = _sample(logits, temperature=temperature, top_k=top_k)
        codes.append(sampled)
        if codebook == speech.config.num_codebooks - 1:
            break
        embedding = depth.embed_code(sampled, codebook)
        logits = _depth_scores(
            depth.step(embedding, codebook_index=codebook, cache=cache),
            None
            if uncond_cache is None
            else depth.step(embedding, codebook_index=codebook, cache=uncond_cache),
            cfg_scale,
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
    values = logits if temperature == 0 else logits / temperature
    if top_k > 0 and top_k < values.shape[-1]:
        cutoff = mx.topk(values, top_k)[..., -1:]
        values = mx.where(values < cutoff, mx.array(-1.0e9, dtype=values.dtype), values)
    token = mx.random.categorical(nn.log_softmax(values, axis=-1))
    return token.reshape((1,))
