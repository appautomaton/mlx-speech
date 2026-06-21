"""Inspect a downloaded NVIDIA RE-USE (SEMamba) checkpoint.

Torch-free: reads ``model.safetensors`` with the safetensors numpy backend and
prints the key count, total parameters, and top-level prefixes. Used as the
Slice 1 verification for the RE-USE MLX port (the SEMamba conversion mirror
lands in a later slice).

    uv run python scripts/convert/reuse_inspect.py --checkpoint models/reuse/original
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from safetensors import safe_open


def inspect(checkpoint_dir: Path) -> dict:
    weights = checkpoint_dir / "model.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"no model.safetensors under {checkpoint_dir}")

    total = 0
    prefixes: collections.Counter = collections.Counter()
    num_keys = 0
    with safe_open(str(weights), "numpy") as f:
        for key in f.keys():
            shape = f.get_slice(key).get_shape()
            count = 1
            for dim in shape:
                count *= dim
            total += count
            prefixes[key.split(".")[0]] += 1
            num_keys += 1

    config_path = checkpoint_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    return {
        "num_keys": num_keys,
        "total_params": total,
        "prefixes": dict(prefixes.most_common()),
        "model_cfg": config.get("model_cfg", {}),
        "stft_cfg": config.get("stft_cfg", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a RE-USE/SEMamba checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/reuse/original"),
        help="Directory containing model.safetensors (and config.json).",
    )
    args = parser.parse_args()

    info = inspect(args.checkpoint)
    print(f"NUM_KEYS {info['num_keys']}")
    print(f"TOTAL_PARAMS {info['total_params']} (~{info['total_params'] / 1e6:.2f}M)")
    print("PREFIXES " + ", ".join(f"{name}={n}" for name, n in info["prefixes"].items()))
    if info["model_cfg"]:
        print("MODEL_CFG " + json.dumps(info["model_cfg"]))
    if info["stft_cfg"]:
        print("STFT_CFG " + json.dumps(info["stft_cfg"]))


if __name__ == "__main__":
    main()
