"""High-level non-stream inference wrapper for Step-Audio-EditX."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
import time

import mlx.core as mx
import numpy as np

from ..models.step_audio_editx import (
    LoadedStepAudioEditXModel,
    LoadedStepAudioFlowConditioner,
    LoadedStepAudioFlowModel,
    LoadedStepAudioHiFTModel,
    StepAudioCosyVoiceFrontEnd,
    StepAudioEditXTokenizer,
    load_step_audio_editx_model,
    load_step_audio_flow_conditioner,
    load_step_audio_flow_model,
    load_step_audio_hift_model,
)
from ..models.step_audio_tokenizer import (
    LoadedStepAudioVQ02Model,
    LoadedStepAudioVQ06Model,
    format_audio_token_string,
    load_step_audio_vq02_model,
    load_step_audio_vq06_model,
    pack_raw_codes_to_prompt_tokens,
    resolve_step_audio_tokenizer_model_dir,
)

_DEFAULT_CLONE_SPEAKER = "debug"
_DEFAULT_TOTAL_SEQUENCE_LIMIT = 8192


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 1:
        return waveform
    if waveform.ndim != 2:
        raise ValueError(f"Expected mono or simple multi-channel audio, got {waveform.shape}.")
    if waveform.shape[0] == 1:
        return waveform[0].astype(np.float32, copy=False)
    if waveform.shape[1] == 1:
        return waveform[:, 0].astype(np.float32, copy=False)
    if waveform.shape[0] <= waveform.shape[1]:
        return waveform.mean(axis=0, dtype=np.float32)
    return waveform.mean(axis=1, dtype=np.float32)


def _cap_peak(waveform: np.ndarray, *, target_peak: float = 0.6) -> np.ndarray:
    peak = float(np.max(np.abs(waveform))) if waveform.size > 0 else 0.0
    if peak <= target_peak or peak <= 0.0:
        return waveform.astype(np.float32, copy=False)
    return (waveform / peak * float(target_peak)).astype(np.float32, copy=False)


def _sample_next_token(logits: mx.array, *, temperature: float) -> int:
    if temperature <= 0.0:
        return int(mx.argmax(logits, axis=-1).item())
    sampled = mx.random.categorical(logits.astype(mx.float32) / float(temperature), axis=-1)
    return int(sampled.item())


def _build_audio_edit_instruction(
    *,
    prompt_text: str,
    edit_type: str,
    edit_info: str | None = None,
    target_text: str | None = None,
) -> str:
    text = prompt_text.strip()
    if edit_type in {"emotion", "speed"}:
        if not edit_info:
            raise ValueError(f"edit_info is required for edit_type={edit_type!r}.")
        if edit_info == "remove":
            return (
                "Remove any emotion in the following audio and the reference text is: "
                f"{text}\n"
            )
        return (
            f"Make the following audio more {edit_info}. "
            f"The text corresponding to the audio is: {text}\n"
        )
    if edit_type == "style":
        if not edit_info:
            raise ValueError("edit_info is required for edit_type='style'.")
        if edit_info == "remove":
            return (
                "Remove any speaking styles in the following audio and the reference text is: "
                f"{text}\n"
            )
        return (
            f"Make the following audio more {edit_info} style. "
            f"The text corresponding to the audio is: {text}\n"
        )
    if edit_type == "denoise":
        return (
            "Remove any noise from the given audio while preserving the voice content clearly. "
            "Ensure that the speech quality remains intact with minimal distortion, and eliminate "
            "all noise from the audio.\n"
        )
    if edit_type == "vad":
        return (
            "Remove any silent portions from the given audio while preserving the voice content "
            "clearly. Ensure that the speech quality remains intact with minimal distortion, and "
            "eliminate all silence from the audio.\n"
        )
    if edit_type == "paralinguistic":
        if not target_text:
            raise ValueError("target_text is required for edit_type='paralinguistic'.")
        return (
            "Add some non-verbal sounds to make the audio more natural, the new text is : "
            f"{target_text}\n  The text corresponding to the audio is: {text}\n"
        )
    raise ValueError(f"Unsupported Step-Audio edit_type: {edit_type!r}.")


@dataclass(frozen=True)
class StepAudioEditXResult:
    waveform: np.ndarray
    sample_rate: int
    generated_token_ids: list[int]
    generated_step1_token_ids: list[int]
    generated_dual_timesteps: int
    generated_mel_frames: int
    expected_duration_seconds: float
    stop_reached: bool
    stop_reason: str
    mode: str
    elapsed_sec: float | None = None
    rtf: float | None = None


@dataclass
class StepAudioEditXModel:
    """Loaded Step-Audio-EditX runtime ready for non-stream inference."""

    step1: LoadedStepAudioEditXModel
    tokenizer: StepAudioEditXTokenizer
    tokenizer_dir: Path
    vq02: LoadedStepAudioVQ02Model
    vq06: LoadedStepAudioVQ06Model
    frontend: StepAudioCosyVoiceFrontEnd
    conditioner: LoadedStepAudioFlowConditioner
    flow: LoadedStepAudioFlowModel
    hift: LoadedStepAudioHiFTModel

    @classmethod
    def from_dir(
        cls,
        model_dir: str | Path | None = None,
        *,
        tokenizer_dir: str | Path | None = None,
        prefer_mlx_int8: bool = False,
        strict: bool = True,
    ) -> "StepAudioEditXModel":
        step1 = load_step_audio_editx_model(
            model_dir,
            prefer_mlx_int8=prefer_mlx_int8,
            strict=strict,
        )
        resolved_tokenizer_dir = resolve_step_audio_tokenizer_model_dir(tokenizer_dir)
        return cls(
            step1=step1,
            tokenizer=StepAudioEditXTokenizer.from_path(step1.model_dir),
            tokenizer_dir=resolved_tokenizer_dir,
            vq02=load_step_audio_vq02_model(resolved_tokenizer_dir, strict=strict),
            vq06=load_step_audio_vq06_model(resolved_tokenizer_dir, strict=strict),
            frontend=StepAudioCosyVoiceFrontEnd.from_model_dir(step1.model_dir),
            conditioner=load_step_audio_flow_conditioner(step1.model_dir),
            flow=load_step_audio_flow_model(step1.model_dir),
            hift=load_step_audio_hift_model(step1.model_dir),
        )

    @classmethod
    def from_path(
        cls,
        model_dir: str | Path | None = None,
        *,
        tokenizer_dir: str | Path | None = None,
        prefer_mlx_int8: bool = False,
        strict: bool = True,
    ) -> "StepAudioEditXModel":
        return cls.from_dir(
            model_dir,
            tokenizer_dir=tokenizer_dir,
            prefer_mlx_int8=prefer_mlx_int8,
            strict=strict,
        )

    def clone(
        self,
        prompt_audio: np.ndarray,
        prompt_sample_rate: int,
        prompt_text: str,
        target_text: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        seed: int | None = None,
        flow_steps: int = 10,
    ) -> StepAudioEditXResult:
        start_time = time.perf_counter()
        prompt_ids, prompt_token, prompt_feat, speaker_embedding = self._prepare_clone_inputs(
            prompt_audio=prompt_audio,
            prompt_sample_rate=prompt_sample_rate,
            prompt_text=prompt_text,
            target_text=target_text,
        )
        result = self._synthesize(
            prompt_ids=prompt_ids,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            speaker_embedding=speaker_embedding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            flow_steps=flow_steps,
            mode="clone",
        )
        return self._attach_runtime_metrics(result, elapsed_sec=time.perf_counter() - start_time)

    def edit(
        self,
        prompt_audio: np.ndarray,
        prompt_sample_rate: int,
        prompt_text: str,
        edit_type: str,
        *,
        edit_info: str | None = None,
        target_text: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float = 0.7,
        seed: int | None = None,
        flow_steps: int = 10,
    ) -> StepAudioEditXResult:
        start_time = time.perf_counter()
        prompt_ids, prompt_token, prompt_feat, speaker_embedding = self._prepare_edit_inputs(
            prompt_audio=prompt_audio,
            prompt_sample_rate=prompt_sample_rate,
            prompt_text=prompt_text,
            edit_type=edit_type,
            edit_info=edit_info,
            target_text=target_text,
        )
        result = self._synthesize(
            prompt_ids=prompt_ids,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            speaker_embedding=speaker_embedding,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            flow_steps=flow_steps,
            mode="edit",
        )
        return self._attach_runtime_metrics(result, elapsed_sec=time.perf_counter() - start_time)

    def _prepare_prompt_audio(
        self,
        prompt_audio: np.ndarray,
    ) -> np.ndarray:
        waveform = _to_mono_float32(prompt_audio)
        if waveform.size == 0:
            raise ValueError("prompt_audio must not be empty.")
        return _cap_peak(waveform)

    def _prepare_clone_inputs(
        self,
        *,
        prompt_audio: np.ndarray,
        prompt_sample_rate: int,
        prompt_text: str,
        target_text: str,
    ) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
        waveform = self._prepare_prompt_audio(prompt_audio)
        prompt_vq02 = self.vq02.runtime.encode(waveform, int(prompt_sample_rate))
        prompt_vq06 = self.vq06.runtime.encode(waveform, int(prompt_sample_rate))
        prompt_wav_tokens = format_audio_token_string(prompt_vq02, prompt_vq06)
        prompt_ids = self.tokenizer.build_clone_prompt_ids(
            speaker=_DEFAULT_CLONE_SPEAKER,
            prompt_text=prompt_text,
            prompt_wav_tokens=prompt_wav_tokens,
            target_text=target_text,
        )
        prompt_token = pack_raw_codes_to_prompt_tokens(prompt_vq02, prompt_vq06)
        prompt_feat, _ = self.frontend.extract_speech_feat(waveform, int(prompt_sample_rate))
        speaker_embedding = self.frontend.extract_spk_embedding(waveform, int(prompt_sample_rate))
        return prompt_ids, prompt_token, prompt_feat, speaker_embedding

    def _prepare_edit_inputs(
        self,
        *,
        prompt_audio: np.ndarray,
        prompt_sample_rate: int,
        prompt_text: str,
        edit_type: str,
        edit_info: str | None,
        target_text: str | None,
    ) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
        waveform = self._prepare_prompt_audio(prompt_audio)
        prompt_vq02 = self.vq02.runtime.encode(waveform, int(prompt_sample_rate))
        prompt_vq06 = self.vq06.runtime.encode(waveform, int(prompt_sample_rate))
        audio_token_str = format_audio_token_string(prompt_vq02, prompt_vq06)
        prompt_ids = self.tokenizer.build_edit_prompt_ids(
            instruct_prefix=_build_audio_edit_instruction(
                prompt_text=prompt_text,
                edit_type=edit_type,
                edit_info=edit_info,
                target_text=target_text,
            ),
            audio_token_str=audio_token_str,
        )
        prompt_token = pack_raw_codes_to_prompt_tokens(prompt_vq02, prompt_vq06)
        prompt_feat, _ = self.frontend.extract_speech_feat(waveform, int(prompt_sample_rate))
        speaker_embedding = self.frontend.extract_spk_embedding(waveform, int(prompt_sample_rate))
        return prompt_ids, prompt_token, prompt_feat, speaker_embedding

    def _resolve_audio_token_base_id(self) -> int:
        return self.tokenizer.token_to_id("<audio_0>")

    def _resolve_max_new_tokens(self, prompt_length: int, max_new_tokens: int | None) -> int:
        model_limit = int(self.step1.config.max_seq_len) - int(prompt_length)
        if model_limit <= 0:
            raise ValueError(
                f"Prompt length {prompt_length} exceeds the Step1 context limit "
                f"{self.step1.config.max_seq_len}."
            )
        if max_new_tokens is not None:
            if max_new_tokens <= 0:
                raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}.")
            return min(int(max_new_tokens), model_limit)
        default_limit = _DEFAULT_TOTAL_SEQUENCE_LIMIT - int(prompt_length)
        if default_limit > 0:
            return min(default_limit, model_limit)
        return model_limit

    def _attach_runtime_metrics(
        self,
        result: StepAudioEditXResult,
        *,
        elapsed_sec: float,
    ) -> StepAudioEditXResult:
        elapsed = max(float(elapsed_sec), 0.0)
        audio_duration = (
            float(result.waveform.shape[0]) / float(result.sample_rate)
            if result.sample_rate > 0
            else 0.0
        )
        rtf = (audio_duration / elapsed) if elapsed > 0.0 and audio_duration > 0.0 else None
        return replace(
            result,
            elapsed_sec=elapsed,
            rtf=rtf,
        )

    def _generate_audio_tokens(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int | None,
        temperature: float,
        seed: int | None,
    ) -> tuple[list[int], list[int], bool, str]:
        if seed is not None:
            mx.random.seed(int(seed))

        generation_limit = self._resolve_max_new_tokens(len(prompt_ids), max_new_tokens)
        audio_token_base_id = self._resolve_audio_token_base_id()
        eos_token_id = int(self.step1.config.eos_token_id)

        outputs = self.step1.model(
            input_ids=mx.array([prompt_ids], dtype=mx.int32),
        )
        mx.eval(outputs.logits)
        cache = outputs.cache
        next_logits = outputs.logits[:, -1, :]

        generated_token_ids: list[int] = []
        generated_step1_token_ids: list[int] = []
        stop_reached = False
        stop_reason = "max_new_tokens"

        for _ in range(generation_limit):
            next_token = _sample_next_token(next_logits[0], temperature=temperature)
            generated_step1_token_ids.append(int(next_token))
            if next_token == eos_token_id:
                stop_reached = True
                stop_reason = "eos"
                break
            if next_token >= audio_token_base_id:
                generated_token_ids.append(int(next_token - audio_token_base_id))
            outputs = self.step1.model(
                input_ids=mx.array([[next_token]], dtype=mx.int32),
                cache=cache,
            )
            mx.eval(outputs.logits)
            cache = outputs.cache
            next_logits = outputs.logits[:, -1, :]

        return generated_token_ids, generated_step1_token_ids, stop_reached, stop_reason

    def _synthesize(
        self,
        *,
        prompt_ids: list[int],
        prompt_token: list[int],
        prompt_feat: np.ndarray,
        speaker_embedding: np.ndarray,
        max_new_tokens: int | None,
        temperature: float,
        seed: int | None,
        flow_steps: int,
        mode: str,
    ) -> StepAudioEditXResult:
        generated_token_ids, generated_step1_token_ids, stop_reached, stop_reason = self._generate_audio_tokens(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
        )
        if not generated_token_ids:
            raise RuntimeError(
                "Step-Audio did not generate any audio token ids; stop_reason="
                f"{stop_reason!r}."
            )

        prepared = self.conditioner.model.prepare_nonstream_inputs(
            token=generated_token_ids,
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            speaker_embedding=speaker_embedding,
        )
        mel = self.flow.model.inference(prepared, n_timesteps=int(flow_steps))
        waveform, _ = self.hift.model.inference(mel)
        mono = np.asarray(waveform[0], dtype=np.float32)
        if mono.ndim != 1:
            raise ValueError(f"Expected a mono waveform from HiFT, got shape {mono.shape}.")
        generated_dual_timesteps = int(prepared.token_dual.shape[1])
        generated_mel_frames = int(mel.shape[2])
        vocoder_hop_samples = int(
            math.prod(self.hift.config.upsample_rates) * self.hift.config.istft_hop_len
        )
        expected_duration_seconds = (
            float(generated_mel_frames * vocoder_hop_samples) / float(self.hift.config.sampling_rate)
        )

        return StepAudioEditXResult(
            waveform=mono,
            sample_rate=int(self.hift.config.sampling_rate),
            generated_token_ids=generated_token_ids,
            generated_step1_token_ids=generated_step1_token_ids,
            generated_dual_timesteps=generated_dual_timesteps,
            generated_mel_frames=generated_mel_frames,
            expected_duration_seconds=expected_duration_seconds,
            stop_reached=stop_reached,
            stop_reason=stop_reason,
            mode=mode,
        )


__all__ = [
    "StepAudioEditXModel",
    "StepAudioEditXResult",
]
