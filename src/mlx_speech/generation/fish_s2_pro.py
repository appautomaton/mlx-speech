from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from ..models.fish_s2_pro import (
    Conversation,
    DualARTransformer,
    FishS2Codec,
    FishS2ProConfig,
    FishS2Tokenizer,
    Message,
    TextPart,
)
from ..models.fish_s2_pro.checkpoint import (
    load_checkpoint_into_model,
    load_fish_s2_pro_checkpoint,
)


_DEFAULT_TEMPERATURE = 1.0
_DEFAULT_TOP_P = 0.9
_DEFAULT_TOP_K = 30
_RAS_WINDOW = 10
_RAS_HIGH_TEMPERATURE = 1.0
_RAS_HIGH_TOP_P = 0.9


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

    def _stop_token_id_set(self) -> set[int]:
        return {int(self.tokenizer.im_end_id), int(self.config.eos_token_id)}

    def _semantic_token_id_set(self) -> set[int]:
        semantic_ids = {
            int(token_id) for token_id in self.tokenizer.semantic_token_ids.values()
        }
        return semantic_ids

    @staticmethod
    def _resolve_codec_dir(
        model_dir: str | Path,
        codec_dir: str | Path | None = None,
    ) -> Path:
        if codec_dir is not None:
            return Path(codec_dir)

        resolved = Path(model_dir)
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
    ) -> FishS2ProOutput:
        conversation = Conversation(
            [Message(role="user", modality="voice", parts=[TextPart(text)])]
        )
        prompt = conversation.encode_for_inference(
            self.tokenizer,
            self.config.audio_decoder_config.num_codebooks,
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
        inverse_semantic_ids = {
            semantic_token_id: code
            for code, semantic_token_id in self.tokenizer.semantic_token_ids.items()
        }
        if token_id not in inverse_semantic_ids:
            raise ValueError(
                f"Selected token id {token_id} is not EOS and not a valid semantic token"
            )
        return int(inverse_semantic_ids[token_id])

    def _apply_semantic_logit_bias(self, logits: mx.array) -> mx.array:
        vocab_size = int(logits.shape[-1])
        valid = mx.zeros((vocab_size,), dtype=mx.bool_)
        for token_id in self._semantic_token_id_set() | self._stop_token_id_set():
            if 0 <= token_id < vocab_size:
                valid[token_id] = True
        blocked = mx.full(logits.shape, float("-inf"), dtype=logits.dtype)
        return mx.where(valid[None, :], logits, blocked)

    def _generate_codes(
        self,
        prompt: mx.array,
        *,
        max_new_tokens: int = 256,
    ) -> mx.array:
        generated = []
        cur = prompt[None, :, :]
        num_codebooks = self.config.audio_decoder_config.num_codebooks
        recent_semantic_token_ids: list[int] = []

        for _ in range(max_new_tokens):
            forward = self.model(cur)
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
            next_semantic = mx.array([next_semantic_token_id], dtype=mx.int32)
            if next_semantic_token_id in self._stop_token_id_set():
                break

            semantic_code = self._semantic_code_from_token_id(next_semantic_token_id)
            recent_semantic_token_ids.append(next_semantic_token_id)
            recent_semantic_token_ids = recent_semantic_token_ids[-_RAS_WINDOW:]
            previous_codebooks = mx.zeros((1, 0), dtype=mx.int32)
            step_codes = [mx.array([semantic_code], dtype=mx.int32)]

            for _codebook_idx in range(1, num_codebooks):
                fast_logits = self.model.fast_forward(
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
            step_codebooks = mx.concatenate(
                [mx.array([[semantic_code]], dtype=mx.int32), previous_codebooks],
                axis=1,
            )

            next_frame = mx.zeros((1, num_codebooks + 1, 1), dtype=mx.int32)
            next_frame[:, 0, 0] = next_semantic
            next_frame[:, 1:, 0] = step_codebooks
            cur = mx.concatenate([cur, next_frame], axis=2)

        if not generated:
            return mx.zeros((num_codebooks, 0), dtype=mx.int32)

        return mx.stack(generated, axis=1).squeeze(2)


def generate_fish_s2_pro(
    text: str,
    *,
    model_dir: str = "models/fish_s2_pro/original",
    codec_dir: str | None = None,
    max_new_tokens: int = 256,
) -> FishS2ProOutput:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    runtime = FishS2ProRuntime.from_dir(model_dir, codec_dir=codec_dir)
    if max_new_tokens == 256:
        return runtime.synthesize(text)
    return runtime.synthesize(text, max_new_tokens=max_new_tokens)
