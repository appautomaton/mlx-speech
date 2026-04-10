"""CLI entry point for mlx-speech TTS generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate speech with mlx-speech TTS models.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available TTS models and exit.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path, alias (e.g. fish-s2-pro), or HF repo ID.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to synthesize.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.wav",
        help="Output WAV path (default: output.wav).",
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        help="Reference audio path for voice cloning.",
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help="Transcript of reference audio for voice cloning.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--codec",
        default=None,
        help="Codec model path or HF repo (for MOSS models).",
    )
    return parser.parse_args()


def _print_models() -> None:
    from . import list_models

    print("Available TTS models:\n")
    for alias, (repo, desc) in list_models().items():
        print(f"  {alias:<16} {desc}")
        print(f"  {' ' * 16} {repo}\n")


def main() -> None:
    args = parse_args()

    if args.list_models:
        _print_models()
        return

    if not args.model or not args.text:
        print("Error: --model and --text are required.", file=sys.stderr)
        sys.exit(1)

    if (args.reference_audio is None) != (args.reference_text is None):
        print(
            "Error: --reference-audio and --reference-text must both be provided.",
            file=sys.stderr,
        )
        sys.exit(1)

    from . import load
    from ..audio import write_wav

    model = load(args.model, codec_path_or_repo=args.codec)

    generate_kwargs = {}
    if args.reference_audio is not None:
        generate_kwargs["reference_audio"] = args.reference_audio
        generate_kwargs["reference_text"] = args.reference_text
    if args.max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = args.max_new_tokens

    result = model.generate(args.text, **generate_kwargs)

    output_path = write_wav(Path(args.output), result.waveform, sample_rate=result.sample_rate)
    print(f"Wrote {output_path} (sample_rate={result.sample_rate})")


if __name__ == "__main__":
    main()
