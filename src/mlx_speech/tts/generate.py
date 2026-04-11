"""TTS generation command handler.

The public entry point is :func:`tts_main` which takes pre-parsed argparse
namespace. A standalone :func:`main` is also provided so
``python -m mlx_speech.tts.generate`` still works.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_tts_args(parser: argparse.ArgumentParser) -> None:
    """Register TTS CLI arguments on the given parser."""
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
        help="Text to synthesize (required for most models; optional in edit mode).",
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
        help="Reference audio path for voice cloning or editing.",
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help="Transcript of reference audio.",
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
    parser.add_argument(
        "--edit-type",
        default=None,
        choices=[
            "emotion",
            "style",
            "speed",
            "denoise",
            "vad",
            "paralinguistic",
        ],
        help="Audio edit type (Step Audio only).",
    )
    parser.add_argument(
        "--edit-info",
        default=None,
        help="Edit qualifier, e.g. 'happy', 'whispering', 'fast'.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Target duration in seconds (Moss Sound Effect).",
    )


def _print_models() -> None:
    from . import list_models

    print("Available TTS models:\n")
    for alias, (repo, desc) in list_models().items():
        print(f"  {alias:<20} {desc}")
        print(f"  {' ' * 20} {repo}\n")


def tts_main(args: argparse.Namespace) -> None:
    """Execute the TTS command with a pre-parsed argparse namespace."""
    if args.list_models:
        _print_models()
        return

    if not args.model:
        print("Error: --model is required.", file=sys.stderr)
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

    generate_kwargs: dict = {}
    if args.reference_audio is not None:
        generate_kwargs["reference_audio"] = args.reference_audio
        generate_kwargs["reference_text"] = args.reference_text
    if args.max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.edit_type is not None:
        generate_kwargs["edit_type"] = args.edit_type
    if args.edit_info is not None:
        generate_kwargs["edit_info"] = args.edit_info
    if args.duration_seconds is not None:
        generate_kwargs["duration_seconds"] = args.duration_seconds

    result = model.generate(args.text, **generate_kwargs)

    output_path = write_wav(
        Path(args.output), result.waveform, sample_rate=result.sample_rate
    )
    print(f"Wrote {output_path} (sample_rate={result.sample_rate})")


def main() -> None:
    """Standalone entry point for ``python -m mlx_speech.tts.generate``."""
    parser = argparse.ArgumentParser(
        description="Generate speech with mlx-speech TTS models.",
    )
    add_tts_args(parser)
    args = parser.parse_args()
    tts_main(args)


if __name__ == "__main__":
    main()
