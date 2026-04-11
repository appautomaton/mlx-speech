"""Unified mlx-speech CLI entry point.

Usage:

    mlx-speech tts --model fish-s2-pro --text "Hello" -o out.wav
    mlx-speech asr --model cohere-asr --audio input.wav
    mlx-speech tts --list-models
    mlx-speech --help
"""

from __future__ import annotations

import argparse
import sys


def _build_parser() -> argparse.ArgumentParser:
    from .asr.generate import add_asr_args
    from .tts.generate import add_tts_args

    parser = argparse.ArgumentParser(
        prog="mlx-speech",
        description="MLX-native speech library for Apple Silicon.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    tts_parser = subparsers.add_parser(
        "tts",
        help="Generate speech from text or audio (TTS, voice cloning, editing, sound effects).",
    )
    add_tts_args(tts_parser)

    asr_parser = subparsers.add_parser(
        "asr",
        help="Transcribe audio to text.",
    )
    add_asr_args(asr_parser)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "tts":
        from .tts.generate import tts_main

        tts_main(args)
    elif args.command == "asr":
        from .asr.generate import asr_main

        asr_main(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
