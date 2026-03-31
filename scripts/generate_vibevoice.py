"""Smoke test for VibeVoice Large generation.

Usage:
    python scripts/generate_vibevoice.py --text "Hello world"
    python scripts/generate_vibevoice.py --text "Hello" --reference-audio ref.wav -o out.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mlx.core as mx

from mlx_speech.audio.io import load_audio, write_wav
from mlx_speech.generation.vibevoice import (
    VibeVoiceGenerationConfig,
    synthesize_vibevoice,
)
from mlx_speech.models.vibevoice.checkpoint import load_vibevoice_model
from mlx_speech.models.vibevoice.tokenizer import VibeVoiceTokenizer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VibeVoice Large TTS smoke test")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--model-dir", default=None, help="Model checkpoint directory")
    parser.add_argument("--tokenizer-dir", default=None, help="Tokenizer directory")
    parser.add_argument("--reference-audio", default=None, help="Reference audio for cloning")
    parser.add_argument("-o", "--output", default="outputs/vibevoice_test.wav")
    parser.add_argument("--cfg-scale", type=float, default=1.3)
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=20,
        help="Base diffusion step count. Used for all frames unless --diffusion-steps-fast is set.",
    )
    parser.add_argument(
        "--diffusion-steps-fast",
        type=int,
        default=None,
        help="Optional reduced diffusion step count used after the warmup window.",
    )
    parser.add_argument(
        "--diffusion-warmup-frames",
        type=int,
        default=10,
        help="Number of early generated frames that keep the full --diffusion-steps budget.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature. Ignored when --greedy is set.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling filter. Ignored when --greedy is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampled generation.",
    )
    parser.add_argument("--greedy", action="store_true")
    return parser


def _build_generation_config(args: argparse.Namespace) -> VibeVoiceGenerationConfig:
    return VibeVoiceGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        cfg_scale=args.cfg_scale,
        diffusion_steps=args.diffusion_steps,
        diffusion_steps_fast=args.diffusion_steps_fast,
        diffusion_warmup_frames=args.diffusion_warmup_frames,
        do_sample=not args.greedy,
        temperature=0.0 if args.greedy else args.temperature,
        top_p=args.top_p,
        seed=args.seed,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir or 'default'}...")
    t0 = time.perf_counter()
    loaded = load_vibevoice_model(args.model_dir, strict=False)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  alignment exact: {loaded.alignment_report.is_exact_match}")
    if not loaded.alignment_report.is_exact_match:
        print(f"  missing in model: {len(loaded.alignment_report.missing_in_model)}")
        print(f"  missing in ckpt:  {len(loaded.alignment_report.missing_in_checkpoint)}")
        print(f"  shape mismatches: {len(loaded.alignment_report.shape_mismatches)}")
        if loaded.alignment_report.missing_in_model:
            for k in loaded.alignment_report.missing_in_model[:5]:
                print(f"    ckpt-only: {k}")
        if loaded.alignment_report.missing_in_checkpoint:
            for k in loaded.alignment_report.missing_in_checkpoint[:5]:
                print(f"    model-only: {k}")
        if loaded.alignment_report.shape_mismatches:
            for k, ms, cs in loaded.alignment_report.shape_mismatches[:5]:
                print(f"    shape: {k} model={ms} ckpt={cs}")

    tokenizer_dir = args.tokenizer_dir or args.model_dir or str(loaded.model_dir)
    print(f"Loading tokenizer from {tokenizer_dir}...")
    tokenizer = VibeVoiceTokenizer.from_path(tokenizer_dir)
    print(f"  speech_start_id={tokenizer.speech_start_id}")
    print(f"  speech_end_id={tokenizer.speech_end_id}")
    print(f"  speech_diffusion_id={tokenizer.speech_diffusion_id}")
    print(f"  eos_token_id={tokenizer.eos_token_id}")

    # Load reference audio if provided
    ref_audio = None
    if args.reference_audio:
        print(f"Loading reference audio: {args.reference_audio}")
        ref_waveform, sr = load_audio(
            args.reference_audio, sample_rate=24000, mono=True,
        )
        ref_audio = ref_waveform.reshape(1, 1, -1)  # (1, 1, T)
        print(f"  {ref_audio.shape[2]} samples at 24000 Hz")

    config = _build_generation_config(args)

    print(f"Generating: '{args.text}'")
    t0 = time.perf_counter()
    result = synthesize_vibevoice(
        loaded.model, tokenizer, args.text,
        reference_audio=ref_audio,
        config=config,
    )
    # Force evaluation to get accurate timing
    mx.eval(result.waveform)
    gen_time = time.perf_counter() - t0

    print(f"  generated_tokens: {result.generated_tokens}")
    print(f"  stop_reached: {result.stop_reached}")
    print(f"  waveform samples: {result.waveform.shape[0]}")
    audio_dur = result.waveform.shape[0] / result.sample_rate
    print(f"  audio duration: {audio_dur:.2f}s")
    print(f"  generation time: {gen_time:.2f}s")
    if audio_dur > 0:
        print(f"  RTF: {audio_dur / gen_time:.2f}x")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, result.waveform, sample_rate=result.sample_rate)
    print(f"  saved: {output_path}")


if __name__ == "__main__":
    main()
