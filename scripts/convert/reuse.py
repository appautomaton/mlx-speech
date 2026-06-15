"""Convert the NVIDIA RE-USE (SEMamba) checkpoint to pure-MLX safetensors.

Torch-free: reads the upstream ``model.safetensors`` with the safetensors numpy
backend, remaps every key onto the MLX `SEMamba` parameter tree (see
`mlx_speech.models.reuse.loader`), verifies the key sets match exactly, and
writes ``model.safetensors`` for the MLX runtime.

    uv run python scripts/convert/reuse.py \
        --input-dir models/reuse/original --output-dir models/reuse/mlx

Weights are NSCLv1 (non-commercial) and never committed; ``models/`` is
gitignored.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_speech.models.reuse.loader import assert_keys_match, build_mlx_state
from mlx_speech.models.reuse.semamba import SEMamba

DEFAULT_INPUT = Path("models/reuse/original")
DEFAULT_OUTPUT = Path("models/reuse/mlx")


def convert(input_dir: Path, output_dir: Path) -> dict[str, int]:
    """Convert and write MLX weights; return a small summary dict."""
    state = build_mlx_state(input_dir)

    model = SEMamba()
    model_keys = {k for k, _ in tree_flatten(model.parameters())}
    assert_keys_match(model_keys, set(state))

    mx.eval(list(state.values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(output_dir / "model.safetensors"), state, metadata={"format": "mlx"}
    )

    config = input_dir / "config.json"
    if config.is_file():
        shutil.copy2(config, output_dir / "config.json")

    return {"num_keys": len(state)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert NVIDIA RE-USE / SEMamba to MLX."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    summary = convert(args.input_dir, args.output_dir)
    print(f"Converted RE-USE/SEMamba: {summary['num_keys']} keys -> {args.output_dir}")


if __name__ == "__main__":
    main()
