"""Single-pass forced alignment for Qwen3-ForcedAligner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..models.qwen3_asr.config import Qwen3ASRConfig
from ..models.qwen3_asr.feature_extraction import Qwen3ASRFeatureBatch, Qwen3ASRFeatureExtractor
from ..models.qwen3_asr.forced_aligner import (
    MAX_ALIGN_SECONDS,
    TIMESTAMP_TOKEN,
    Qwen3ForcedAlignerSettings,
    build_aligner_prompt,
    decode_timestamps,
    split_text_units,
    validate_align_language,
)
from ..models.qwen3_asr.model import Qwen3ASRModel
from ..models.qwen3_asr.text_decoder import FAST_DECODE_COMPUTE
from ..models.qwen3_asr.tokenizer import Qwen3ASRTokenizer


SAMPLE_RATE = 16000


@dataclass(frozen=True)
class Qwen3ForcedAlignInputs:
    units: tuple[str, ...]
    input_ids: tuple[int, ...]
    timestamp_positions: tuple[int, ...]
    features: Qwen3ASRFeatureBatch
    duration_s: float


@dataclass(frozen=True)
class Qwen3ForcedAlignResult:
    spans: tuple[tuple[str, float, float], ...]
    timestamp_classes: tuple[int, ...]
    duration_s: float
    prompt_tokens: int


@dataclass
class Qwen3ForcedAligner:
    model: Qwen3ASRModel
    tokenizer: Qwen3ASRTokenizer
    feature_extractor: Qwen3ASRFeatureExtractor
    config: Qwen3ASRConfig
    settings: Qwen3ForcedAlignerSettings

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path,
        *,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "Qwen3ForcedAligner":
        from ..models.qwen3_asr.checkpoint import (
            get_quantization_config,
            load_checkpoint_into_model,
            load_qwen3_asr_checkpoint,
            quantize_qwen3_asr_model,
        )

        model_dir = Path(model_dir)
        # Reject non-aligner packages before reading weights.
        settings = Qwen3ForcedAlignerSettings.from_config(Qwen3ASRConfig.from_dir(model_dir))
        checkpoint = load_qwen3_asr_checkpoint(model_dir)
        model = Qwen3ASRModel(checkpoint.config)
        quantization = get_quantization_config(checkpoint.config)
        if quantization is not None:
            quantize_qwen3_asr_model(model, quantization, state_dict=checkpoint.state_dict)
        load_checkpoint_into_model(model, checkpoint, strict=True)
        model.set_dtype(dtype)
        model.eval()
        mx.eval(model.parameters())

        tokenizer = Qwen3ASRTokenizer.from_dir(model_dir, qwen2_pretokenizer=True)
        timestamp_id = tokenizer.token_to_id(TIMESTAMP_TOKEN)
        if timestamp_id != settings.timestamp_token_id:
            raise ValueError(
                f"{TIMESTAMP_TOKEN} id mismatch: tokenizer={timestamp_id}, "
                f"config={settings.timestamp_token_id}."
            )
        return cls(
            model=model,
            tokenizer=tokenizer,
            feature_extractor=Qwen3ASRFeatureExtractor.from_dir(model_dir),
            config=checkpoint.config,
            settings=settings,
        )

    def prepare(
        self,
        audio: np.ndarray | mx.array | str | Path,
        text: str,
        *,
        language: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> Qwen3ForcedAlignInputs:
        validate_align_language(language)
        units = split_text_units(text)
        if not units:
            raise ValueError(
                "Transcript has nothing to align: it needs letters, numbers, or CJK characters."
            )
        waveform = _load_waveform(audio, sample_rate=sample_rate)
        duration_s = waveform.shape[0] / SAMPLE_RATE
        if duration_s > MAX_ALIGN_SECONDS:
            raise ValueError(
                f"Audio is {duration_s:.1f} s; Qwen3-ForcedAligner accepts at most "
                f"{MAX_ALIGN_SECONDS:.0f} s. Split it and align each part."
            )

        features = self.feature_extractor(waveform, sample_rate=SAMPLE_RATE)
        audio_length = int(features.audio_lengths.reshape(-1)[0])
        prompt = build_aligner_prompt(
            units,
            audio_length=audio_length,
            audio_bos_token=self.tokenizer.audio_bos_token,
            audio_token=self.tokenizer.audio_token,
            audio_eos_token=self.tokenizer.audio_eos_token,
        )
        input_ids = tuple(int(i) for i in self.tokenizer.encode(prompt))
        timestamp_id = self.settings.timestamp_token_id
        positions = tuple(i for i, token in enumerate(input_ids) if token == timestamp_id)
        if len(positions) != 2 * len(units):
            raise RuntimeError(
                f"Aligner prompt has {len(positions)} timestamp tokens for {len(units)} units."
            )
        limit = self.config.text_config.max_position_embeddings
        if len(input_ids) > limit:
            raise ValueError(
                f"Aligner prompt is {len(input_ids)} tokens; the model accepts {limit}. "
                "Shorten the transcript or split the audio."
            )
        return Qwen3ForcedAlignInputs(
            units=tuple(units),
            input_ids=input_ids,
            timestamp_positions=positions,
            features=features,
            duration_s=duration_s,
        )

    def timestamp_hidden_states(self, inputs: Qwen3ForcedAlignInputs) -> mx.array:
        """Final decoder hidden states at the ``<timestamp>`` positions, ``[N, hidden]``."""

        audio_features = self.model.get_audio_features(
            mx.array(inputs.features.input_features, dtype=mx.float32),
            feature_attention_mask=mx.array(inputs.features.feature_attention_mask, dtype=mx.int32),
        )
        input_ids = mx.array([inputs.input_ids], dtype=mx.int32)
        inputs_embeds = self.model.prepare_inputs_embeds(input_ids, audio_features)
        hidden = self.model.text_decoder.model(
            inputs_embeds=inputs_embeds, compute=FAST_DECODE_COMPUTE
        ).last_hidden_state
        return hidden[0, mx.array(inputs.timestamp_positions, dtype=mx.int32), :]

    def align(
        self,
        audio: np.ndarray | mx.array | str | Path,
        text: str,
        *,
        language: str,
        sample_rate: int = SAMPLE_RATE,
    ) -> Qwen3ForcedAlignResult:
        inputs = self.prepare(audio, text, language=language, sample_rate=sample_rate)
        logits = self.model.text_decoder.project_logits(self.timestamp_hidden_states(inputs))
        classes = np.array(mx.argmax(logits, axis=-1), dtype=np.int64)
        spans = decode_timestamps(
            inputs.units,
            classes,
            segment_time_ms=self.settings.timestamp_segment_time_ms,
        )
        return Qwen3ForcedAlignResult(
            spans=tuple(spans),
            timestamp_classes=tuple(int(c) for c in classes),
            duration_s=inputs.duration_s,
            prompt_tokens=len(inputs.input_ids),
        )


def _load_waveform(
    audio: np.ndarray | mx.array | str | Path,
    *,
    sample_rate: int,
) -> np.ndarray:
    if isinstance(audio, (str, Path)):
        from ..audio import load_audio

        loaded, _ = load_audio(audio, sample_rate=SAMPLE_RATE, mono=True)
        waveform = np.array(loaded, dtype=np.float32).reshape(-1)
    else:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Qwen3-ForcedAligner needs {SAMPLE_RATE} Hz arrays; got {sample_rate} Hz. "
                "Resample first, e.g. with mlx_speech.audio.resample_audio."
            )
        waveform = np.array(audio, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(f"Expected a 1-D mono waveform, got shape {waveform.shape}.")
    if waveform.size == 0:
        raise ValueError("Audio is empty.")
    # Upstream rescales out-of-range input by its peak, then clips.
    peak = float(np.max(np.abs(waveform)))
    if peak > 1.0:
        waveform = waveform / peak
    return np.clip(waveform, -1.0, 1.0)
