#!/usr/bin/env python3
"""Transcribe audio with CohereLabs/cohere-transcribe-03-2026 (MLX).

Examples:
    python scripts/generate/cohere_asr.py --audio speech.wav
    python scripts/generate/cohere_asr.py --audio speech.wav --language fr
    python scripts/generate/cohere_asr.py --audio speech.wav --no-punctuation
    python scripts/generate/cohere_asr.py --audio speech.wav --itn
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np


def _load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    """Load audio from file as float32 mono at target_sr."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except ImportError:
        # Fallback: try scipy.io.wavfile for plain WAV
        from scipy.io import wavfile
        sr, audio = wavfile.read(str(path))
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / 32768.0

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        try:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(target_sr, sr)
            audio = resample_poly(audio, target_sr // g, sr // g).astype(np.float32)
        except ImportError:
            # Fallback: linear interpolation via numpy (adequate quality for speech)
            old_len = len(audio)
            new_len = int(round(old_len * target_sr / sr))
            audio = np.interp(
                np.linspace(0, old_len - 1, new_len),
                np.arange(old_len),
                audio,
            ).astype(np.float32)

    return audio.astype(np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", required=True, help="Path to audio file (WAV/FLAC/MP3/…)")
    p.add_argument(
        "--model-dir",
        default="models/cohere/cohere_transcribe/mlx-int8",
        help="Path to the MLX CohereAsr checkpoint directory.",
    )
    p.add_argument("--language", default="en", help="ISO 639-1 language code (default: en)")
    p.add_argument("--no-punctuation", action="store_true", help="Disable punctuation")
    p.add_argument("--itn", action="store_true", help="Enable inverse text normalization")
    p.add_argument("--max-new-tokens", type=int, default=448)
    p.add_argument("-o", "--output", help="Write transcript to this file instead of stdout")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(
            f"Error: model directory not found: {model_dir}\n"
            "Run scripts/convert/cohere_asr.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Loading model from {model_dir} …", file=sys.stderr)
    from mlx_speech.generation.cohere_asr import CohereAsrModel
    asr = CohereAsrModel.from_dir(model_dir)

    print(f"Loading audio from {audio_path} …", file=sys.stderr)
    audio = _load_audio(audio_path)
    duration_s = len(audio) / 16000
    print(f"  {len(audio)} samples ({duration_s:.1f}s)", file=sys.stderr)

    print("Transcribing …", file=sys.stderr)
    result = asr.transcribe(
        audio,
        language=args.language,
        punctuation=not args.no_punctuation,
        itn=args.itn,
        max_new_tokens=args.max_new_tokens,
    )

    if args.output:
        Path(args.output).write_text(result.text + "\n", encoding="utf-8")
        print(f"Transcript written to {args.output}", file=sys.stderr)
    else:
        print(result.text)


if __name__ == "__main__":
    main()
