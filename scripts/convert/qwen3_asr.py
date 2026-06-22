#!/usr/bin/env python3
"""Package Qwen3-ASR weights for the local MLX runtime.

Produces a bf16 package or a quantized (affine int8 / microscaling mxfp8) package,
following the family-dir convention ``models/qwen3_asr_1_7b/{mlx-bf16,mlx-int8,mlx-mxfp8}``.

Examples:
    python scripts/convert/qwen3_asr.py --quant int8
    python scripts/convert/qwen3_asr.py --quant mxfp8
    python scripts/convert/qwen3_asr.py --quant bf16
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
from mlx_speech.models.qwen3_asr.model import Qwen3ASRModel  # noqa: E402


DEFAULT_INPUT = Path("models/qwen3_asr_1_7b/original")
DEFAULT_BASE_OUTPUT = Path("models/qwen3_asr_1_7b")

# --quant preset -> (output subdir, quantization config or None for bf16).
# mxfp8 is microscaling FP8: group_size MUST be 32 and there is no bias term.
_QUANT_PRESETS: dict[str, tuple[str, QuantizationConfig | None]] = {
    "bf16": ("mlx-bf16", None),
    "int8": ("mlx-int8", QuantizationConfig(bits=8, group_size=64, mode="affine")),
    "mxfp8": ("mlx-mxfp8", QuantizationConfig(bits=8, group_size=32, mode="mxfp8")),
}


def convert_qwen3_asr(
    input_dir: Path,
    output_dir: Path,
    *,
    copy_supporting_files: bool = True,
):
    """Package an unquantized bf16 Qwen3-ASR checkpoint (back-compat helper)."""
    checkpoint = load_qwen3_asr_checkpoint(input_dir)
    return save_qwen3_asr_bf16_checkpoint(
        checkpoint,
        output_dir,
        copy_supporting_files=copy_supporting_files,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--quant",
        choices=sorted(_QUANT_PRESETS),
        default="int8",
        help="Build to produce. Default: int8 (the runtime default).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output dir. Default: models/qwen3_asr_1_7b/mlx-<quant>.",
    )
    parser.add_argument(
        "--skip-supporting-files",
        action="store_true",
        help="Only write model.safetensors; do not copy tokenizer/config assets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subdir, quantization = _QUANT_PRESETS[args.quant]
    output_dir = args.output_dir or (DEFAULT_BASE_OUTPUT / subdir)
    copy_supporting = not args.skip_supporting_files

    print(f"Input:  {args.input_dir}")
    print(f"Output: {output_dir}")
    if quantization is None:
        print("Quant:  bf16 (unquantized)")
    else:
        print(
            f"Quant:  {quantization.mode} {quantization.bits}-bit "
            f"group_size={quantization.group_size}"
        )
    print()

    checkpoint = load_qwen3_asr_checkpoint(args.input_dir)
    print(f"Tensors: {len(checkpoint.state_dict)} after sanitization")
    print(f"Skipped: {len(checkpoint.skipped_keys)}  Renamed: {len(checkpoint.renamed_keys)}")

    if quantization is None:
        report = save_qwen3_asr_bf16_checkpoint(
            checkpoint,
            output_dir,
            copy_supporting_files=copy_supporting,
        )
        print(f"Saved:   {report.output_file}")
        if report.copied_files:
            print(f"Copied:  {len(report.copied_files)} supporting files")
        return

    print("Building model and loading weights (strict)...")
    model = Qwen3ASRModel(checkpoint.config)
    load_checkpoint_into_model(model, checkpoint, strict=True)

    print("Quantizing eligible Linear/Embedding layers...")
    quantize_qwen3_asr_model(model, quantization)

    save_qwen3_asr_model(
        model,
        output_dir,
        config=checkpoint.config,
        quantization=quantization,
        copy_supporting_files_from=args.input_dir if copy_supporting else None,
    )
    print(f"Saved:   {output_dir / 'model.safetensors'}")


if __name__ == "__main__":
    main()
