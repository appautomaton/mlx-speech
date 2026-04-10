"""CLI entry point for mlx-speech ASR transcription."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio with mlx-speech ASR models.",
    )
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
    return parser.parse_args()


def _print_models() -> None:
    from . import list_models

    print("Available ASR models:\n")
    for alias, (repo, desc) in list_models().items():
        print(f"  {alias:<16} {desc}")
        print(f"  {' ' * 16} {repo}\n")


def main() -> None:
    import sys

    args = parse_args()

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


if __name__ == "__main__":
    main()
