from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ..audio import load_audio
from ..models.fish_s2_pro import (
    Conversation,
    DualARTransformer,
    FishS2Codec,
    FishS2ProConfig,
    FishS2Tokenizer,
    Message,
    TextPart,
)
from ..models.fish_s2_pro.prompt import VQPart
from ..models.fish_s2_pro.cache import KVCache
from ..models.fish_s2_pro.checkpoint import (
    load_checkpoint_into_model,
    load_fish_s2_pro_checkpoint,
    quantize_fish_s2_pro_model,
)


_DEFAULT_TEMPERATURE = 0.8
_DEFAULT_TOP_P = 0.8
_DEFAULT_TOP_K = 30
_RAS_WINDOW = 10
_RAS_HIGH_TEMPERATURE = 1.0
_RAS_HIGH_TOP_P = 0.9
_SYSTEM_PROMPT = "convert the provided text to speech"
_CLONE_SYSTEM_PROMPT = (
    "convert the provided text to speech reference to the following:\n\nText:\n"
)


def _apply_top_k(logits: mx.array, top_k: int) -> mx.array:
    if top_k <= 0 or top_k >= int(logits.shape[-1]):
        return logits
    kth_values = mx.topk(logits, k=top_k, axis=-1)[..., -1:]
    neg_inf = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
    return mx.where(logits < kth_values, neg_inf, logits)


def _apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    if top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be > 0.")

    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    to_remove = cumulative_probs > top_p
    to_remove = mx.concatenate(
        [mx.zeros_like(to_remove[..., :1]), to_remove[..., :-1]], axis=-1
    )
    neg_inf = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
    filtered_sorted_logits = mx.where(to_remove, neg_inf, sorted_logits)
    filtered_logits = mx.full(logits.shape, neg_inf, dtype=logits.dtype)
    return mx.put_along_axis(
        filtered_logits, sorted_indices, filtered_sorted_logits, axis=-1
    )


def _sample_token_id(
    logits: mx.array,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    warped = logits.astype(mx.float32)
    if temperature <= 0.0:
        return int(mx.argmax(warped, axis=-1)[0])
    warped = warped / float(temperature)
    warped = _apply_top_k(warped, top_k)
    warped = _apply_top_p(warped, top_p)
    return int(mx.random.categorical(warped, axis=-1)[0])


@dataclass
class FishS2ProOutput:
    waveform: mx.array
    sample_rate: int
    generated_tokens: int


@dataclass
class FishS2ProRuntime:
    model: DualARTransformer
    tokenizer: FishS2Tokenizer
    codec: FishS2Codec
    config: FishS2ProConfig

    def __post_init__(self):
        self._stop_ids = frozenset(
            {int(self.tokenizer.im_end_id), int(self.config.eos_token_id)}
        )
        self._semantic_ids = frozenset(
            int(tid) for tid in self.tokenizer.semantic_token_ids.values()
        )
        self._inverse_semantic = {
            int(tid): int(code)
            for code, tid in self.tokenizer.semantic_token_ids.items()
        }
        self._semantic_bias: mx.array | None = None

    def _build_generation_prompt(
        self,
        text: str,
        *,
        reference_text: str | None = None,
        reference_codes: mx.array | None = None,
    ) -> mx.array:
        if reference_text is not None and reference_codes is not None:
            system_parts = [
                TextPart(_CLONE_SYSTEM_PROMPT),
                TextPart(f"<|speaker:0|>{reference_text}"),
                TextPart("\n\nSpeech:\n"),
                VQPart(reference_codes),
            ]
        else:
            system_parts = [TextPart(_SYSTEM_PROMPT)]

        conversation = Conversation(
            [
                Message(role="system", parts=system_parts),
                Message(role="user", parts=[TextPart(text)]),
                Message(
                    role="assistant",
                    modality="voice",
                    parts=[],
                    add_im_start=True,
                    add_im_end=False,
                ),
            ]
        )
        return conversation.encode_for_inference(
            self.tokenizer,
            self.config.audio_decoder_config.num_codebooks,
        )

    @staticmethod
    def _resolve_codec_dir(
        model_dir: str | Path,
        codec_dir: str | Path | None = None,
    ) -> Path:
        if codec_dir is not None:
            return Path(codec_dir)

        resolved = Path(model_dir)
        bundled_codec_dir = resolved / "codec-mlx"
        if bundled_codec_dir.is_dir():
            return bundled_codec_dir
        sibling_codec_dir = resolved.parent / "codec-mlx"
        if sibling_codec_dir.is_dir():
            return sibling_codec_dir
        return resolved

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        codec_dir: str | Path | None = None,
    ) -> "FishS2ProRuntime":
        resolved = Path(model_dir)
        checkpoint = load_fish_s2_pro_checkpoint(resolved)
        model = DualARTransformer(checkpoint.config)
        if checkpoint.config.quantization is not None:
            quantize_fish_s2_pro_model(
                model,
                checkpoint.config.quantization,
                state_dict=checkpoint.state_dict,
            )
        load_checkpoint_into_model(model, checkpoint, strict=True)
        tokenizer = FishS2Tokenizer.from_pretrained(str(resolved))
        codec = FishS2Codec.from_dir(cls._resolve_codec_dir(resolved, codec_dir))
        return cls(
            model=model,
            tokenizer=tokenizer,
            codec=codec,
            config=checkpoint.config,
        )

    def synthesize(
        self,
        text: str,
        *,
        max_new_tokens: int = 256,
        reference_audio: str | Path | None = None,
        reference_text: str | None = None,
    ) -> FishS2ProOutput:
        if (reference_audio is None) != (reference_text is None):
            raise ValueError(
                "reference_audio and reference_text must both be provided for voice cloning"
            )

        reference_codes = None
        if reference_audio is not None:
            waveform, _sr = load_audio(
                reference_audio, sample_rate=self.codec.sample_rate, mono=True
            )
            reference_codes = self.codec.encode(waveform)

        prompt = self._build_generation_prompt(
            text,
            reference_text=reference_text,
            reference_codes=reference_codes,
        )
        codes = self._generate_codes(prompt, max_new_tokens=max_new_tokens)
        if int(codes.shape[1]) == 0:
            raise ValueError("No Fish S2 audio tokens generated before stop token")
        waveform = self.codec.decode(codes)
        return FishS2ProOutput(
            waveform=waveform.squeeze(),
            sample_rate=self.codec.sample_rate,
            generated_tokens=int(codes.shape[1]),
        )

    def _semantic_code_from_token_id(self, token_id: int) -> int:
        if token_id not in self._inverse_semantic:
            raise ValueError(
                f"Selected token id {token_id} is not EOS and not a valid semantic token"
            )
        return self._inverse_semantic[token_id]

    def _get_semantic_bias(self, vocab_size: int) -> mx.array:
        if self._semantic_bias is not None and int(self._semantic_bias.shape[0]) == vocab_size:
            return self._semantic_bias
        bias = mx.full((vocab_size,), float("-inf"), dtype=mx.float32)
        for token_id in self._semantic_ids | self._stop_ids:
            if 0 <= token_id < vocab_size:
                bias[token_id] = 0.0
        mx.eval(bias)
        self._semantic_bias = bias
        return bias

    def _apply_semantic_logit_bias(self, logits: mx.array) -> mx.array:
        return logits + self._get_semantic_bias(int(logits.shape[-1]))

    def _make_cache(self, prompt_len: int, max_new_tokens: int) -> KVCache:
        tc = self.config.text_config
        return KVCache(
            num_layers=tc.n_layer,
            dim=tc.dim,
            max_length=prompt_len + max_new_tokens,
        )

    def _generate_codes(
        self,
        prompt: mx.array,
        *,
        max_new_tokens: int = 256,
    ) -> mx.array:
        num_codebooks = self.config.audio_decoder_config.num_codebooks
        cache = self._make_cache(int(prompt.shape[1]), max_new_tokens)

        # Prefill: process entire prompt, populate KV cache
        forward = self.model(prompt[None, :, :], cache=cache)
        mx.eval(forward.logits, forward.hidden_states)

        # Compile the fast AR forward — pure computation, no cache state
        _compiled_fast = (
            mx.compile(self.model.fast_forward)
            if isinstance(self.model, DualARTransformer)
            else self.model.fast_forward
        )

        generated = []
        recent_semantic_token_ids: list[int] = []

        for _ in range(max_new_tokens):
            semantic_logits = forward.logits[:, -1, :]
            semantic_logits = self._apply_semantic_logit_bias(semantic_logits)
            next_semantic_token_id = _sample_token_id(
                semantic_logits,
                temperature=_DEFAULT_TEMPERATURE,
                top_p=_DEFAULT_TOP_P,
                top_k=_DEFAULT_TOP_K,
            )
            fallback_semantic_token_id = _sample_token_id(
                semantic_logits,
                temperature=_RAS_HIGH_TEMPERATURE,
                top_p=_RAS_HIGH_TOP_P,
                top_k=_DEFAULT_TOP_K,
            )
            if next_semantic_token_id in recent_semantic_token_ids:
                next_semantic_token_id = fallback_semantic_token_id
            if next_semantic_token_id in self._stop_ids:
                break

            semantic_code = self._semantic_code_from_token_id(next_semantic_token_id)
            recent_semantic_token_ids.append(next_semantic_token_id)
            recent_semantic_token_ids = recent_semantic_token_ids[-_RAS_WINDOW:]
            previous_codebooks = mx.array([[semantic_code]], dtype=mx.int32)
            step_codes = [mx.array([semantic_code], dtype=mx.int32)]

            for _codebook_idx in range(1, num_codebooks):
                fast_logits = _compiled_fast(
                    forward.hidden_states[:, -1:, :], previous_codebooks
                )
                next_codebook = mx.array(
                    [
                        _sample_token_id(
                            fast_logits,
                            temperature=_DEFAULT_TEMPERATURE,
                            top_p=_DEFAULT_TOP_P,
                            top_k=_DEFAULT_TOP_K,
                        )
                    ],
                    dtype=mx.int32,
                )
                previous_codebooks = mx.concatenate(
                    [previous_codebooks, next_codebook[:, None]], axis=1
                )
                step_codes.append(next_codebook)

            generated.append(mx.stack(step_codes, axis=0))

            # Decode: single-token forward with cached KV
            next_frame = mx.zeros((1, num_codebooks + 1, 1), dtype=mx.int32)
            next_frame[:, 0, 0] = mx.array([next_semantic_token_id], dtype=mx.int32)
            next_frame[:, 1:, 0] = previous_codebooks
            forward = self.model(next_frame, cache=cache)
            mx.eval(forward.logits, forward.hidden_states)

        if not generated:
            return mx.zeros((num_codebooks, 0), dtype=mx.int32)

        return mx.stack(generated, axis=1).squeeze(2)


def generate_fish_s2_pro(
    text: str,
    *,
    model_dir: str = "models/fish_s2_pro/original",
    codec_dir: str | None = None,
    max_new_tokens: int = 256,
    reference_audio: str | None = None,
    reference_text: str | None = None,
) -> FishS2ProOutput:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    runtime = FishS2ProRuntime.from_dir(model_dir, codec_dir=codec_dir)
    return runtime.synthesize(
        text,
        max_new_tokens=max_new_tokens,
        reference_audio=reference_audio,
        reference_text=reference_text,
    )
