"""CLI for `DramaBoxModel.generate(...)`. Writes a 48 kHz WAV to disk.

Example:

    .venv/bin/python scripts/generate_dramabox.py \
        --dramabox-dir models/dramabox/mlx-bf16 \
        --gemma-dir models/gemma_3_12b_it_backbone/mlx-4bit \
        --prompt 'A woman speaks clearly.' \
        --voice-ref outputs/source/hank_hill_ref.wav \
        --duration 5.0 \
        --out outputs/dramabox_smoke.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import soundfile as sf

from mlx_speech.generation.dramabox import DramaBoxModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with DramaBox v5.")
    parser.add_argument("--dramabox-dir", type=Path, required=True,
                        help="Path containing dramabox-dit-v1.safetensors and dramabox-audio-components.safetensors")
    parser.add_argument("--gemma-dir", type=Path, required=True,
                        help="Path to the MLX 4-bit Gemma 3 12B IT directory")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output WAV path")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Duration in seconds")
    parser.add_argument("--voice-ref", type=Path, default=None,
                        help="Optional reference audio for transcript-free voice cloning")
    parser.add_argument("--denoise-ref", action="store_true",
                        help="Reserved placeholder; currently raises because RE-USE is deferred")
    parser.add_argument("--cfg-scale", type=float, default=2.5)
    parser.add_argument("--stg-scale", type=float, default=0.0,
                        help="Non-zero requires STG perturbation in DiT block "
                             "(v5 baseline falls back to CFG-only)")
    parser.add_argument("--rescale-scale", type=str, default="auto",
                        help="'auto' or a float")
    parser.add_argument("--modality-scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"loading DramaBox from {args.dramabox_dir}")
    model = DramaBoxModel.from_dir(args.dramabox_dir, gemma_dir=args.gemma_dir)

    rescale = args.rescale_scale
    try:
        rescale = float(rescale)
    except ValueError:
        pass  # 'auto'

    print(f"generating {args.duration:.1f}s for prompt: {args.prompt!r}")
    result = model.generate(
        args.prompt,
        duration_s=args.duration,
        cfg_scale=args.cfg_scale,
        stg_scale=args.stg_scale,
        rescale_scale=rescale,
        modality_scale=args.modality_scale,
        steps=args.steps,
        seed=args.seed,
        voice_ref=args.voice_ref,
        denoise_ref=args.denoise_ref,
    )

    # waveform: [2, T_samples] fp32 in [-1, 1] → write as stereo WAV
    wf = np.array(result.waveform, copy=False)  # mx.array → numpy
    # soundfile expects (T, C) for stereo
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out), wf.T, result.sample_rate, subtype="FLOAT")

    print(f"wrote {args.out} ({wf.shape[1]} samples @ {result.sample_rate} Hz, {wf.shape[1] / result.sample_rate:.2f}s)")


if __name__ == "__main__":
    main()
