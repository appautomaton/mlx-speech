"""ASR transcription command handler.

The public entry point is :func:`asr_main` which takes a pre-parsed argparse
namespace. A standalone :func:`main` is also provided so
``python -m mlx_speech.asr.generate`` still works.
"""

from __future__ import annotations

import argparse
import sys


def add_asr_args(parser: argparse.ArgumentParser) -> None:
    """Register ASR CLI arguments on the given parser."""
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available ASR models and exit.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path, alias (e.g. cohere-asr), or HF repo ID.",
    )
    parser.add_argument(
        "--audio",
        default=None,
        help="Audio file to transcribe.",
    )
    parser.add_argument(
        "--language",
        default="en",
        help="Language code (default: en).",
    )


def _print_models() -> None:
    from . import list_models

    print("Available ASR models:\n")
    for alias, (repo, desc) in list_models().items():
        print(f"  {alias:<20} {desc}")
        print(f"  {' ' * 20} {repo}\n")


def asr_main(args: argparse.Namespace) -> None:
    """Execute the ASR command with a pre-parsed argparse namespace."""
    if args.list_models:
        _print_models()
        return

    if not args.model or not args.audio:
        print("Error: --model and --audio are required.", file=sys.stderr)
        sys.exit(1)

    from . import load

    model = load(args.model)
    result = model.generate(args.audio, language=args.language)
    print(result.text)


def main() -> None:
    """Standalone entry point for ``python -m mlx_speech.asr.generate``."""
    parser = argparse.ArgumentParser(
        description="Transcribe audio with mlx-speech ASR models.",
    )
    add_asr_args(parser)
    args = parser.parse_args()
    asr_main(args)


if __name__ == "__main__":
    main()
