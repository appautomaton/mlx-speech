#!/usr/bin/env python3
"""Run local VibeVoice Large inference with MLX and save a WAV file."""

from __future__ import annotations

import argparse
import textwrap
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


class _HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    """Show defaults while preserving example formatting."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run VibeVoice Large inference with local MLX checkpoints.\n\n"
            "Supports single-speaker generation, multi-speaker dialogue (up to 4 speakers),\n"
            "and voice cloning from a reference WAV."
        ),
        epilog=textwrap.dedent(
            """\
            Text format:
              Speaker labels are 0-based: Speaker 0:, Speaker 1:, Speaker 2:, Speaker 3:
              Plain text (no label) is treated as a single Speaker 0: utterance.
              Multi-speaker: one label per line, labels may interleave freely.

            Examples:

              Single speaker:
                python scripts/generate/vibevoice.py \\
                  --text "The quick brown fox jumped over the lazy dog." \\
                  --output outputs/single.wav

              Two speakers, two rounds each:
                python scripts/generate/vibevoice.py \\
                  --text "Speaker 0: Have you tried VibeVoice yet?
            Speaker 1: Not yet. Does it need PyTorch?
            Speaker 0: No. Pure MLX, runs locally on Apple Silicon.
            Speaker 1: That is impressive. I had no idea." \\
                  --output outputs/dialogue.wav

              Voice clone (single speaker):
                python scripts/generate/vibevoice.py \\
                  --text "Speaker 0: This voice was cloned from the reference." \\
                  --reference-audio-speaker0 outputs/source/ref.wav \\
                  --output outputs/clone.wav

              Voice clone (two speakers, each with a reference):
                python scripts/generate/vibevoice.py \\
                  --text "Speaker 0: Hey, welcome to the show.
            Speaker 1: Great to be here." \\
                  --reference-audio-speaker0 outputs/source/speaker0_ref.wav \\
                  --reference-audio-speaker1 outputs/source/speaker1_ref.wav \\
                  --output outputs/clone_dialogue.wav

            Sampling mode:
              Default is deterministic (greedy), matching upstream behaviour.
              Add --no-greedy to enable temperature + top-p sampling.

            Diffusion quality vs speed:
              --diffusion-steps 20    default, matches upstream
              More steps = theoretically better quality but slower (upstream max: 100).
              The quality/step tradeoff has not been benchmarked for the MLX int8 runtime.

            Model loading:
              Default local runtime: models/vibevoice/mlx-int8
              Use --model-dir to point at any other checkpoint directory.
            """
        ),
        formatter_class=_HelpFormatter,
    )
    parser.add_argument(
        "--text",
        required=True,
        help=(
            "Text to synthesize. Use 'Speaker 0:', 'Speaker 1:', ... prefixes for "
            "multi-speaker dialogue (0-based). Plain text defaults to Speaker 0."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="VibeVoice checkpoint directory. Defaults to models/vibevoice/mlx-int8.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=None,
        help="Tokenizer directory. Defaults to --model-dir.",
    )
    for spk_idx in range(4):
        parser.add_argument(
            f"--reference-audio-speaker{spk_idx}",
            default=None,
            help=(
                f"Reference WAV for Speaker {spk_idx} voice cloning. "
                "Provide one per speaker to clone each voice independently. "
                "Speakers without a reference get a model-default voice."
            ),
        )
    parser.add_argument(
        "-o", "--output",
        default="outputs/vibevoice_out.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.3,
        help="Classifier-free guidance scale for the diffusion head.",
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=20,
        help="Diffusion denoising steps per audio frame. Higher = better quality, slower.",
    )
    parser.add_argument(
        "--diffusion-steps-fast",
        type=int,
        default=None,
        help=(
            "Reduced diffusion step count used after --diffusion-warmup-frames. "
            "Speeds up generation after the initial quality window."
        ),
    )
    parser.add_argument(
        "--diffusion-warmup-frames",
        type=int,
        default=10,
        help="Number of early frames that use the full --diffusion-steps budget before switching to --diffusion-steps-fast.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum number of speech tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.95,
        help="LM sampling temperature. Only used when --no-greedy is set.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling filter. Only used when --no-greedy is set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed. Upstream default is 42.",
    )
    parser.add_argument(
        "--no-greedy",
        action="store_true",
        help="Enable sampling mode (temperature + top-p). Upstream default is deterministic/greedy.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _build_generation_config(args: argparse.Namespace) -> VibeVoiceGenerationConfig:
    return VibeVoiceGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        cfg_scale=args.cfg_scale,
        diffusion_steps=args.diffusion_steps,
        diffusion_steps_fast=args.diffusion_steps_fast,
        diffusion_warmup_frames=args.diffusion_warmup_frames,
        do_sample=args.no_greedy,
        temperature=args.temperature if args.no_greedy else 0.0,
        top_p=args.top_p,
        seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    loaded = load_vibevoice_model(args.model_dir, strict=False)
    tokenizer_dir = args.tokenizer_dir or args.model_dir or str(loaded.model_dir)
    tokenizer = VibeVoiceTokenizer.from_path(tokenizer_dir)

    # Collect per-speaker reference audio in speaker order (skip missing speakers).
    voice_samples: list[mx.array] = []
    ref_paths: list[str] = []
    for spk_idx in range(4):
        path = getattr(args, f"reference_audio_speaker{spk_idx}")
        if path is not None:
            waveform, _ = load_audio(path, sample_rate=24000, mono=True)
            voice_samples.append(waveform.reshape(1, 1, -1))
            ref_paths.append(f"speaker{spk_idx}={path}")

    config = _build_generation_config(args)
    result = synthesize_vibevoice(
        loaded.model, tokenizer, args.text,
        voice_samples=voice_samples if voice_samples else None,
        config=config,
    )
    mx.eval(result.waveform)
    elapsed_sec = time.perf_counter() - start_time

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(output_path, result.waveform, sample_rate=result.sample_rate)

    audio_dur = result.waveform.shape[0] / result.sample_rate

    print("Generated VibeVoice sample")
    print(f"  output: {output_path}")
    print(f"  model_dir: {loaded.model_dir}")
    print(f"  quantized: {'yes' if loaded.quantization is not None else 'no'}")
    print(f"  reference_audio: {', '.join(ref_paths) if ref_paths else 'none'}")
    print(f"  diffusion_steps: {config.diffusion_steps}")
    print(f"  sample_rate: {result.sample_rate}")
    print(f"  generated_tokens: {result.generated_tokens}")
    print(f"  stop_reached: {result.stop_reached}")
    print(f"  audio_duration: {audio_dur:.2f}s")
    print(f"  elapsed_sec: {elapsed_sec:.2f}")
    if audio_dur > 0:
        print(f"  RTF: {audio_dur / elapsed_sec:.2f}x")


if __name__ == "__main__":
    main()
