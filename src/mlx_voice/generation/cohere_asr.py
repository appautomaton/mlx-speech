"""Greedy inference for CohereAsr (encoder-decoder ASR)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from ..models.cohere_asr import (
    CohereAsrForConditionalGeneration,
    CohereAsrFeatureExtractor,
    CohereAsrTokenizer,
    load_cohere_asr_checkpoint,
    load_checkpoint_into_model,
    quantize_cohere_asr_model,
)
from ..models.cohere_asr.checkpoint import get_quantization_config

_NO_SPACE_LANGS = frozenset({"ja", "zh"})


@dataclass(frozen=True)
class CohereAsrResult:
    text: str
    tokens: list[int]
    language: str


@dataclass
class CohereAsrModel:
    """Loaded CohereAsr model ready for inference."""

    model: CohereAsrForConditionalGeneration
    feature_extractor: CohereAsrFeatureExtractor
    tokenizer: CohereAsrTokenizer
    config: Any  # CohereAsrConfig

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "CohereAsrModel":
        model_dir = Path(model_dir)
        ckpt = load_cohere_asr_checkpoint(model_dir)
        model = CohereAsrForConditionalGeneration(ckpt.config)
        # If the checkpoint was quantized, apply quantization structure first
        quant = get_quantization_config(ckpt.config)
        if quant is not None:
            quantize_cohere_asr_model(model, quant, state_dict=ckpt.state_dict)
        load_checkpoint_into_model(model, ckpt, strict=True)
        model.set_dtype(dtype)
        model.eval()  # BatchNorm must use running stats, not batch stats
        mx.eval(model.parameters())

        feature_extractor = CohereAsrFeatureExtractor.from_dir(model_dir)
        tokenizer = CohereAsrTokenizer.from_dir(model_dir)
        return cls(
            model=model,
            feature_extractor=feature_extractor,
            tokenizer=tokenizer,
            config=ckpt.config,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
        language: str = "en",
        punctuation: bool = True,
        max_new_tokens: int = 448,
    ) -> CohereAsrResult:
        """Transcribe a single audio waveform.

        Args:
            audio:        float32 waveform, shape (N,)
            sample_rate:  input sample rate (resampled to 16 kHz if needed)
            language:     ISO 639-1 language code (see tokenizer.LANGUAGES)
            punctuation:  include punctuation in the output
            max_new_tokens: maximum number of tokens to generate

        Returns:
            CohereAsrResult with text, token ids, and language
        """
        if sample_rate != 16000:
            raise ValueError(
                f"CohereAsr requires 16 kHz audio; got {sample_rate} Hz. "
                "Resample before calling transcribe()."
            )

        chunks = self.feature_extractor.process_audio(audio)
        chunk_results: list[CohereAsrResult] = []
        for features, attention_mask in chunks:
            features_mx = mx.array(features)[None]  # (1, T, n_mels)
            if attention_mask is not None:
                mask_mx = mx.array(attention_mask, dtype=mx.bool_)[None]  # (1, T)
            else:
                mask_mx = None
            chunk_results.append(
                self._decode(
                    features_mx,
                    mask_mx,
                    language=language,
                    punctuation=punctuation,
                    max_new_tokens=max_new_tokens,
                )
            )

        if len(chunk_results) == 1:
            return chunk_results[0]

        separator = "" if language in _NO_SPACE_LANGS else " "
        text = separator.join(
            [part for part in [r.text.strip() for r in chunk_results] if part]
        )
        tokens = [token for result in chunk_results for token in result.tokens]
        return CohereAsrResult(text=text, tokens=tokens, language=language)

    def _decode(
        self,
        features: mx.array,
        attention_mask: mx.array | None,
        *,
        language: str,
        punctuation: bool,
        max_new_tokens: int,
    ) -> CohereAsrResult:
        dec_cfg = self.config.decoder

        # --- Encode ---
        encoder_states, encoder_mask = self.model.encode(features, attention_mask)
        mx.eval(encoder_states)

        # --- Build decoder prompt ---
        # Reference: decoder_input_ids = get_decoder_prompt_ids(...) directly.
        # The ▁ token that starts the prompt IS the decoder start token; do not prepend it again.
        prompt_ids = self.tokenizer.get_decoder_prompt_ids(language, punctuation)
        prompt_tensor = mx.array([prompt_ids], dtype=mx.int32)  # (1, L_prompt)

        # --- Prefill ---
        # Run the full prompt through the decoder in one shot to build KV caches.
        logits, self_kvs, cross_kvs = self.model.decode_step(
            prompt_tensor,
            encoder_states,
            encoder_mask,
            self_kv_caches=None,
            cross_kv_caches=None,
            position_offset=0,
        )
        mx.eval(logits)

        # Take the last position's logits as the next-token prediction
        next_token = int(logits[0, -1].argmax())
        generated: list[int] = [next_token]
        position_offset = len(prompt_ids)

        # --- Autoregressive decode ---
        for _ in range(max_new_tokens - 1):
            if next_token == dec_cfg.eos_token_id:
                break

            token_tensor = mx.array([[next_token]], dtype=mx.int32)  # (1, 1)
            logits, self_kvs, cross_kvs = self.model.decode_step(
                token_tensor,
                encoder_states,
                encoder_mask,
                self_kv_caches=self_kvs,
                cross_kv_caches=cross_kvs,
                position_offset=position_offset,
            )
            mx.eval(logits)

            next_token = int(logits[0, 0].argmax())
            generated.append(next_token)
            position_offset += 1

        # Strip trailing EOS if present
        if generated and generated[-1] == dec_cfg.eos_token_id:
            generated = generated[:-1]

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return CohereAsrResult(text=text, tokens=generated, language=language)
