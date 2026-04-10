#!/usr/bin/env python3
"""Generate speech with Fish Audio S2 Pro and save a WAV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mlx_speech.audio import normalize_peak, trim_leading_silence, write_wav
from mlx_speech.generation.fish_s2_pro import generate_fish_s2_pro


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument(
        "--output",
        "-o",
        default="outputs/fish_s2_pro.wav",
        help="Output WAV path.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/fish_s2_pro/original",
        help="Local MLX model directory.",
    )
    parser.add_argument(
        "--codec-dir",
        default=None,
        help="Optional local converted Fish codec directory. Defaults to a sibling codec-mlx directory when present.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--trim-leading-silence",
        action="store_true",
        help="Trim leading low-energy audio before writing the WAV.",
    )
    parser.add_argument(
        "--normalize-peak",
        type=float,
        default=0.0,
        help="Target peak amplitude for output normalization. Set <= 0 to disable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    generation_kwargs = {
        "model_dir": args.model_dir,
        "codec_dir": args.codec_dir,
        "max_new_tokens": args.max_new_tokens,
    }

    result = generate_fish_s2_pro(args.text, **generation_kwargs)

    if result.waveform is None:
        print("Error: No waveform generated", file=sys.stderr)
        sys.exit(1)

    waveform = result.waveform
    if args.trim_leading_silence:
        waveform = trim_leading_silence(
            waveform,
            sample_rate=result.sample_rate,
        )
    if args.normalize_peak > 0:
        waveform = normalize_peak(
            waveform,
            target_peak=args.normalize_peak,
        )

    output_path = write_wav(
        Path(args.output),
        waveform,
        sample_rate=result.sample_rate,
    )

    print("Generated Fish S2 Pro sample")
    print(f"  output: {output_path}")
    print(f"  sample_rate: {result.sample_rate}")
    print(f"  generated_tokens: {result.generated_tokens}")


if __name__ == "__main__":
    main()
