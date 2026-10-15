#!/usr/bin/env python3
"""Generate speech with Fish Audio Drama 3 and save a WAV file.

Speakers are addressed in the text with <|speaker:0|> .. <|speaker:15|>. Repeat
--reference-audio / --reference-text (and optionally --reference-speaker) once per
voice, in the same order.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx_speech
from mlx_speech.audio import write_wav


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--output", "-o", default="outputs/fish_drama_3.wav")
    parser.add_argument("--model-dir", default="models/fish_drama_3/mlx-int8")
    parser.add_argument("--reference-audio", action="append", default=[])
    parser.add_argument("--reference-text", action="append", default=[])
    parser.add_argument(
        "--reference-speaker",
        action="append",
        type=int,
        default=[],
        help="Speaker id (0-15) of each reference; defaults to its transcript tag or 0.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Target length in seconds; generates the text in one pass.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Frames per text chunk, about 21.5 per second (default 1024).",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--single-chunk",
        action="store_true",
        help="Generate the whole text in one pass instead of ~300-byte chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = mlx_speech.tts.load(args.model_dir)
    started = time.perf_counter()
    result = model.generate(
        args.text,
        reference_audio=args.reference_audio or None,
        reference_text=args.reference_text or None,
        reference_speaker=args.reference_speaker or None,
        duration_seconds=args.duration,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        single_chunk=args.single_chunk,
    )
    elapsed = time.perf_counter() - started
    output = write_wav(
        Path(args.output), result.waveform, sample_rate=result.sample_rate
    )
    seconds = result.waveform.shape[-1] / result.sample_rate
    print(f"Generated Fish Audio Drama 3 sample: {output}")
    print(
        f"  audio {seconds:.2f}s in {elapsed:.2f}s ({seconds / elapsed:.2f}x real time)"
    )


if __name__ == "__main__":
    main()
