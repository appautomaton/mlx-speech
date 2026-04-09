#!/usr/bin/env python3
"""Convert Fish Audio S2 Pro to MLX.

Usage:
    python scripts/convert_fish_s2_pro.py
    python scripts/convert_fish_s2_pro.py --input-dir models/fish_s2_pro/original
    python scripts/convert_fish_s2_pro.py --quantize 8
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from safetensors import safe_open
import numpy as np


DEFAULT_INPUT = "models/fish_s2_pro/original"
DEFAULT_OUTPUT = "models/fish_s2_pro/mlx-int8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--quantize", type=int, default=8, choices=[8, 16])
    return parser.parse_args()


def quantize_to_int8(tensor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantize to int8 with per-tensor scale."""
    scale = np.abs(tensor).max() / 127.0
    if scale > 0:
        quantized = (tensor / scale).round().astype(np.int8)
    else:
        quantized = tensor.astype(np.int8)
    return quantized, np.array([scale], dtype=np.float32)


def convert_fish_s2_pro(input_dir: str, output_dir: str, quantize: int = 8) -> None:
    """Convert Fish S2 Pro checkpoint to MLX."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        print(f"Expected: {DEFAULT_INPUT}")
        print("\nPlease download the model first:")
        print(
            "  huggingface-cli download fishaudio/s2-pro --local-dir models/fish_s2_pro/original"
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Bits:   {quantize}")

    # Find all safetensors files
    safetensor_files = list(input_dir.glob("*.safetensors"))
    if not safetensor_files:
        print(f"No safetensors files found in {input_dir}")
        return

    print(f"Found {len(safetensor_files)} files")

    # Process each file
    for fp in sorted(safetensor_files):
        print(f"  Converting {fp.name}...")

        # Read and convert
        state_dict = {}
        with safe_open(fp, framework="np") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if quantize == 8:
                    quantized, scale = quantize_to_int8(tensor)
                    state_dict[key] = quantized
                    state_dict[key + "_scale"] = scale
                else:
                    state_dict[key] = tensor

        # Save (MLX reads numpysafetensors)
        output_file = output_dir / fp.name
        import mlx.core as mx

        mx.save(output_file, state_dict)

        print(f"    Saved {fp.name}")

    # Copy tokenizer files
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
    for tf in tokenizer_files:
        src = input_dir / tf
        dst = output_dir / tf
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {tf}")

    # Copy or create config
    config_src = input_dir / "config.json"
    config_dst = output_dir / "config.json"
    if config_src.exists():
        shutil.copy2(config_src, config_dst)
        print(f"  Copied config.json")
    else:
        print("  Note: No config.json found")

    print("\nDone!")
    print(f"\nTo generate speech:")
    print(f"  python scripts/generate_fish_s2_pro.py --text 'Hello' --output hello.wav")


def main() -> None:
    args = parse_args()
    convert_fish_s2_pro(args.input_dir, args.output_dir, args.quantize)


if __name__ == "__main__":
    main()
