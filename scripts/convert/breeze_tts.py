#!/usr/bin/env python3
"""Package Breeze TTS 2 for MLX.

Main weights stay BF16. The bundled audio tokenizer stays FP32, with
convolution kernels stored in MLX layout. The unused Mimi ``codec_model``
weights are omitted. Original files are left in place.

    python scripts/convert/breeze_tts.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_INPUT = Path("models/breezeblue/breeze_tts_2/original")
DEFAULT_OUTPUT = Path("models/breezeblue/breeze_tts_2/mlx-bf16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    from mlx_speech.models.breeze_tts.checkpoint import convert_breeze_checkpoint

    report = convert_breeze_checkpoint(args.input_dir, args.output_dir)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {report.output_dir}")
    print(f"Main tensors: {report.main_tensors} ({report.tied_audio_embedding} materialized)")
    print(
        "Codec tensors: "
        f"{report.codec_tensors}  transposed: {report.transposed_codec_tensors}"
    )


if __name__ == "__main__":
    main()
