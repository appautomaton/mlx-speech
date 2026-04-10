"""Generation helpers for LongCat AudioDiT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..audio.io import load_audio, write_wav
from ..models.longcat_audiodit.text import approx_duration_from_text, normalize_text
from ..models.longcat_audiodit.tokenizer import LongCatTokenizer


@dataclass(frozen=True)
class LongCatBatchItem:
    uid: str
    prompt_text: str
    prompt_wav_path: Path
    gen_text: str


@dataclass(frozen=True)
class LongCatSynthesisOutput:
    waveform: mx.array
    sample_rate: int
    latent_frames: int


def build_longcat_full_text(
    text: str, prompt_text: str | None = None, *, batch_mode: bool = False
) -> str:
    normalized_text = normalize_text(text)
    if not prompt_text:
        return normalized_text
    normalized_prompt = normalize_text(prompt_text)
    if batch_mode:
        sep = " " if normalized_prompt.endswith(".") else ""
    else:
        sep = " "
    return f"{normalized_prompt}{sep}{normalized_text}"


def parse_longcat_batch_manifest_line(
    line: str, *, line_number: int
) -> LongCatBatchItem:
    parts = line.rstrip("\n").split("|", 3)
    if len(parts) != 4:
        raise ValueError(
            f"Malformed LongCat batch manifest line {line_number}: {line.rstrip()}"
        )
    uid, prompt_text, prompt_wav_path, gen_text = parts
    return LongCatBatchItem(
        uid=uid,
        prompt_text=prompt_text,
        prompt_wav_path=Path(prompt_wav_path),
        gen_text=gen_text,
    )


def synthesize_longcat_audiodit(
    *,
    model,
    tokenizer: LongCatTokenizer,
    text: str,
    prompt_text: str | None = None,
    prompt_audio: mx.array | None = None,
    nfe: int = 16,
    guidance_method: str = "cfg",
    guidance_strength: float = 4.0,
    duration: int | None = None,
    batch_mode: bool = False,
    seed: int | None = 1024,
    initial_noise: mx.array | None = None,
) -> LongCatSynthesisOutput:
    if prompt_audio is not None and not prompt_text:
        raise ValueError("prompt_text is required when prompt_audio is provided")

    normalized_text = normalize_text(text)
    full_text = build_longcat_full_text(text, prompt_text, batch_mode=batch_mode)
    encoded = tokenizer.encode_text([full_text])

    if duration is None:
        sr = model.config.sampling_rate
        latent_hop = model.config.latent_hop
        max_duration = float(model.config.max_wav_duration)
        if prompt_audio is not None and prompt_text:
            prompt_latent, prompt_dur = model.encode_prompt_audio(prompt_audio)
            del prompt_latent
            prompt_time = prompt_dur * latent_hop / sr
            dur_sec = approx_duration_from_text(
                normalized_text, max_duration=max_duration - prompt_time
            )
            approx_pd = approx_duration_from_text(
                normalize_text(prompt_text), max_duration=max_duration
            )
            ratio = float(np.clip(prompt_time / max(approx_pd, 1e-6), 1.0, 1.5))
            dur_sec *= ratio
            duration = int(dur_sec * sr // latent_hop)
            duration = min(duration + prompt_dur, int(max_duration * sr // latent_hop))
        else:
            duration = int(
                approx_duration_from_text(normalized_text, max_duration=max_duration)
                * sr
                // latent_hop
            )

    result = model(
        input_ids=mx.array(encoded["input_ids"], dtype=mx.int32),
        attention_mask=mx.array(encoded["attention_mask"], dtype=mx.int32),
        prompt_audio=prompt_audio,
        duration=duration,
        steps=nfe,
        cfg_strength=guidance_strength,
        guidance_method=guidance_method,
        seed=seed,
        initial_noise=initial_noise,
    )
    waveform = result.waveform.squeeze()
    return LongCatSynthesisOutput(
        waveform=waveform,
        sample_rate=int(model.config.sampling_rate),
        latent_frames=int(result.latent.shape[-1]),
    )


def generate_longcat_audiodit(
    *,
    text: str,
    output_audio: str,
    model_dir: str | None = None,
    prompt_text: str | None = None,
    prompt_audio_path: str | None = None,
    tokenizer_dir: str | None = None,
    nfe: int = 16,
    guidance_method: str = "cfg",
    guidance_strength: float = 4.0,
    seed: int = 1024,
) -> LongCatSynthesisOutput:
    from ..models.longcat_audiodit.checkpoint import (
        load_longcat_model,
        resolve_longcat_tokenizer_dir,
    )

    loaded = load_longcat_model(model_dir=model_dir)
    tokenizer = LongCatTokenizer.from_path(resolve_longcat_tokenizer_dir(tokenizer_dir))
    prompt_audio = None
    if prompt_audio_path is not None:
        waveform, _ = load_audio(
            prompt_audio_path, sample_rate=loaded.config.sampling_rate, mono=True
        )
        prompt_audio = waveform[None, None, :]
    output = synthesize_longcat_audiodit(
        model=loaded.model,
        tokenizer=tokenizer,
        text=text,
        prompt_text=prompt_text,
        prompt_audio=prompt_audio,
        nfe=nfe,
        guidance_method=guidance_method,
        guidance_strength=guidance_strength,
        seed=seed,
    )
    write_wav(output_audio, output.waveform, sample_rate=output.sample_rate)
    return output
