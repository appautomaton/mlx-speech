"""Offline Breeze TTS 2 generation for one voice-design request.

No classifier-free guidance. Sampling stays on device except for the chosen
token id, which the loop needs in order to stop.
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
    decoder_weights = [
        (codec_parameter_key(key), value)
        for key, value in codec_weights.items()
        if key.startswith("decoder.")
    ]
    codec.load_weights(decoder_weights, strict=True)
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
    max_frames: int = 80,
    temperature: float = 0.9,
    top_k: int = 50,
    seed: int = 0,
) -> BreezeAudio:
    """Generate one utterance from a plain ``[speaker]text`` prompt."""

    if max_frames <= 0:
        raise ValueError("max_frames must be positive.")
    mx.random.seed(seed)
    prompt = _prompt_embeddings(speech, tokenizer, f"[{speaker}]{text}")
    cache = speech.backbone_model.make_cache(batch_size=1, dtype=prompt.dtype)
    hidden = speech.backbone_model(prompt, cache)[:, -1, :]
    frames: list[mx.array] = []
    for _ in range(max_frames):
        logits = _mask_reserved(speech.lm_head(hidden), keep_eos=True)
        first = _sample(logits, temperature=temperature, top_k=top_k)
        if int(first.item()) == speech.config.backbone_eos_id:
            break
        frame = _depth_frame(
            speech, hidden, first, temperature=temperature, top_k=top_k
        )
        frames.append(frame)
        hidden = speech.backbone_model(
            speech.backbone_model.embed_audio(frame.reshape(1, 1, -1)), cache
        )[:, -1, :]
    if not frames:
        return BreezeAudio(mx.zeros((0,), dtype=mx.float32), SAMPLE_RATE, 0)
    codes = mx.stack(frames, axis=0)[None, :, :]
    audio = codec.decode(mx.transpose(codes, (0, 2, 1)))
    waveform = audio[0, 0].astype(mx.float32)
    mx.eval(waveform)
    return BreezeAudio(waveform, SAMPLE_RATE, len(frames))


def _prompt_embeddings(
    speech: BreezeSpeech, tokenizer: Tokenizer, text: str
) -> mx.array:
    token_ids = tokenizer.encode(text, add_special_tokens=False).ids
    bos_id = tokenizer.token_to_id("<bos>")
    if bos_id is not None and (not token_ids or token_ids[0] != bos_id):
        token_ids = [bos_id, *token_ids]
    hidden = speech.text_encoder(mx.array(token_ids, dtype=mx.int32)[None, :])
    return speech.text_encoder_proj(hidden)


def _depth_frame(
    speech: BreezeSpeech,
    backbone_hidden: mx.array,
    first: mx.array,
    *,
    temperature: float,
    top_k: int,
) -> mx.array:
    depth = speech.depth_decoder
    cache = depth.make_cache(batch_size=1, dtype=backbone_hidden.dtype)
    codes = [first]
    logits = depth.begin_frame(backbone_hidden, depth.embed_code(first, 0), cache)
    for codebook in range(1, speech.config.num_codebooks):
        sampled = _sample(
            _mask_reserved(logits, keep_eos=False), temperature=temperature, top_k=top_k
        )
        codes.append(sampled)
        if codebook == speech.config.num_codebooks - 1:
            break
        logits = depth.step(
            depth.embed_code(sampled, codebook), codebook_index=codebook, cache=cache
        )
    return mx.concatenate([code.reshape((1,)) for code in codes], axis=0)


def _mask_reserved(logits: mx.array, *, keep_eos: bool) -> mx.array:
    masked = logits.astype(mx.float32)
    masked[..., CODEC_CODEBOOK_SIZE:2051] = -1.0e9
    if not keep_eos and masked.shape[-1] > 2051:
        masked = masked[..., :2051]
    return masked


def _sample(logits: mx.array, *, temperature: float, top_k: int) -> mx.array:
    values = logits if temperature == 0 else logits / temperature
    if top_k > 0 and top_k < values.shape[-1]:
        cutoff = mx.topk(values, top_k)[..., -1:]
        values = mx.where(values < cutoff, mx.array(-1.0e9, dtype=values.dtype), values)
    token = mx.random.categorical(nn.log_softmax(values, axis=-1))
    return token.reshape((1,))
