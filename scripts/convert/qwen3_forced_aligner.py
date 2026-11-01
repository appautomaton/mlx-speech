#!/usr/bin/env python3
"""Package Qwen3-ForcedAligner-0.6B weights for the local MLX runtime.

Produces ``models/qwen3_forced_aligner_0_6b/{mlx-bf16,mlx-int8}``. The int8
build is affine 8-bit, group size 64, over the audio tower, embeddings, and
text decoder; the 5000-class time head stays BF16.

Examples:
    python scripts/convert/qwen3_forced_aligner.py --quant int8
    python scripts/convert/qwen3_forced_aligner.py --quant bf16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mlx_speech.models.qwen3_asr.checkpoint import (  # noqa: E402
    QuantizationConfig,
    load_checkpoint_into_model,
    load_qwen3_asr_checkpoint,
    quantize_qwen3_asr_model,
    save_qwen3_asr_bf16_checkpoint,
    save_qwen3_asr_model,
)
from mlx_speech.models.qwen3_asr.forced_aligner import Qwen3ForcedAlignerSettings  # noqa: E402
from mlx_speech.models.qwen3_asr.model import Qwen3ASRModel  # noqa: E402


DEFAULT_INPUT = Path("models/qwen3_forced_aligner_0_6b/original")
DEFAULT_BASE_OUTPUT = Path("models/qwen3_forced_aligner_0_6b")
INT8 = QuantizationConfig(bits=8, group_size=64, mode="affine")
# Measured against BF16 over 1064 timestamps (zh, en, 59 s): quantizing the
# head doubled the softmax-expected time error (median 0.6 -> 1.4 ms, p95
# 5.7 -> 13.6 ms) to save 5 MB. The audio tower and embeddings added no
# measurable error and save about 440 MB, so they are quantized.
INT8_KEEP_UNQUANTIZED: tuple[str, ...] = ("text_decoder.lm_head",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--quant", choices=("bf16", "int8"), default="int8")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: models/qwen3_forced_aligner_0_6b/mlx-<quant>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_BASE_OUTPUT / f"mlx-{args.quant}"
    checkpoint = load_qwen3_asr_checkpoint(args.input_dir)
    Qwen3ForcedAlignerSettings.from_config(checkpoint.config)

    print(f"Input:   {args.input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Tensors: {len(checkpoint.state_dict)} after sanitization")

    if args.quant == "bf16":
        report = save_qwen3_asr_bf16_checkpoint(checkpoint, output_dir)
        print(f"Saved:   {report.output_file} (+{len(report.copied_files)} supporting files)")
        return

    model = Qwen3ASRModel(checkpoint.config)
    load_checkpoint_into_model(model, checkpoint, strict=True)
    quantize_qwen3_asr_model(model, INT8, keep_unquantized=INT8_KEEP_UNQUANTIZED)
    save_qwen3_asr_model(
        model,
        output_dir,
        config=checkpoint.config,
        quantization=INT8,
        copy_supporting_files_from=args.input_dir,
    )
    kept = ", ".join(INT8_KEEP_UNQUANTIZED) or "nothing"
    print(f"Saved:   {output_dir / 'model.safetensors'} (int8; BF16 kept: {kept})")


if __name__ == "__main__":
    main()
