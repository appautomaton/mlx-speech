#!/usr/bin/env python3
"""Generate non-stream Step-Audio-EditX waveform output locally.

Examples:
    python scripts/generate/step_audio_editx.py clone \
      --prompt-audio ref.wav \
      --prompt-text "Reference transcript." \
      --target-text "New cloned speech." \
      -o outputs/step_audio_clone.wav

    python scripts/generate/step_audio_editx.py edit \
      --prompt-audio ref.wav \
      --prompt-text "Reference transcript." \
      --edit-type denoise \
      -o outputs/step_audio_edit.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_speech.audio.io import load_audio, write_wav
from mlx_speech.generation.step_audio_editx import StepAudioEditXModel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=None, help="Step-Audio-EditX checkpoint directory")
    parser.add_argument("--tokenizer-dir", default=None, help="Step-Audio tokenizer asset directory")
    parser.add_argument(
        "--prefer-mlx-int8",
        action="store_true",
        help="Prefer the mlx-int8 Step1 checkpoint instead of the original checkpoint.",
    )
    parser.add_argument("--prompt-audio", required=True, help="Reference audio file")
    parser.add_argument("--prompt-text", required=True, help="Transcript for the reference audio")
    parser.add_argument("-o", "--output", required=True, help="Output waveform path")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--flow-steps", type=int, default=10)

    subparsers = parser.add_subparsers(dest="mode", required=True)

    clone_parser = subparsers.add_parser("clone", help="Clone speech from a reference audio")
    clone_parser.add_argument("--target-text", required=True, help="Text to synthesize")

    edit_parser = subparsers.add_parser("edit", help="Edit or transform the reference audio")
    edit_parser.add_argument(
        "--edit-type",
        required=True,
        choices=["emotion", "style", "speed", "denoise", "vad", "paralinguistic"],
    )
    edit_parser.add_argument("--edit-info", default=None, help="Edit qualifier (for example: happy, remove, whispering)")
    edit_parser.add_argument(
        "--target-text",
        default=None,
        help="Required only for paralinguistic edits.",
    )

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    prompt_audio, prompt_sample_rate = load_audio(args.prompt_audio, mono=True)
    model = StepAudioEditXModel.from_dir(
        args.model_dir,
        tokenizer_dir=args.tokenizer_dir,
        prefer_mlx_int8=args.prefer_mlx_int8,
    )

    if args.mode == "clone":
        result = model.clone(
            prompt_audio,
            prompt_sample_rate,
            args.prompt_text,
            args.target_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            flow_steps=args.flow_steps,
        )
    else:
        result = model.edit(
            prompt_audio,
            prompt_sample_rate,
            args.prompt_text,
            args.edit_type,
            edit_info=args.edit_info,
            target_text=args.target_text,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed,
            flow_steps=args.flow_steps,
        )

    output_path = write_wav(args.output, result.waveform, sample_rate=result.sample_rate)
    print(f"mode: {result.mode}")
    print(f"sample_rate: {result.sample_rate}")
    print(f"generated_step1_tokens: {len(result.generated_step1_token_ids)}")
    print(f"generated_tokens: {len(result.generated_token_ids)}")
    print(f"generated_dual_timesteps: {result.generated_dual_timesteps}")
    print(f"generated_mel_frames: {result.generated_mel_frames}")
    print(f"expected_duration_seconds: {result.expected_duration_seconds:.2f}")
    if result.elapsed_sec is not None:
        print(f"elapsed_sec: {result.elapsed_sec:.2f}")
    if result.rtf is not None:
        print(f"RTF: {result.rtf:.2f}x")
    print(f"stop_reached: {result.stop_reached}")
    print(f"stop_reason: {result.stop_reason}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
