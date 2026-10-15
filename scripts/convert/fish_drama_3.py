#!/usr/bin/env python3
"""Convert the upstream Fish Audio Drama 3 release into an MLX checkpoint.

Routed experts go from block FP8 to MLX affine quantization, stacked per layer;
every other weight stays bfloat16. The codec is converted without torch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx

from mlx_speech.checkpoints.sharded import INDEX_FILENAME, ShardedCheckpointIndex
from mlx_speech.models.fish_drama_3.checkpoint import (
    EXPERT_PROJECTIONS,
    UPSTREAM_EXPERT_KEY,
    dequantize_fp8_blocks,
    expert_prefix,
    to_mlx_dense,
)
from mlx_speech.models.fish_drama_3.config import ExpertQuantization, FishDrama3Config
from mlx_speech.models.fish_s2_pro.codec_weights import convert_codec_pth_to_assets

DEFAULT_OUTPUT = Path("models/fish_drama_3/mlx-int8")
SHARD_BYTES = 5 * 1024**3
SUPPORTING_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "LICENSE",
)


class ShardWriter:
    """Write tensors into safetensors shards of about shard_bytes, plus an index."""

    def __init__(self, output_dir: Path, shard_bytes: int = SHARD_BYTES):
        self.output_dir = output_dir
        self.shard_bytes = shard_bytes
        self.pending: dict[str, mx.array] = {}
        self.pending_bytes = 0
        self.shards: list[list[str]] = []
        self.written: set[str] = set()
        self.total_bytes = 0

    def add(self, name: str, value: mx.array) -> None:
        if name in self.written:
            raise ValueError(f"{name} written twice")
        self.written.add(name)
        self.pending[name] = value
        self.pending_bytes += value.nbytes
        if self.pending_bytes >= self.shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        path = self.output_dir / f"model-{len(self.shards) + 1:05d}.safetensors"
        mx.save_safetensors(str(path), self.pending, metadata={"format": "mlx"})
        self.shards.append(sorted(self.pending))
        self.total_bytes += self.pending_bytes
        self.pending, self.pending_bytes = {}, 0

    def close(self) -> int:
        """Flush, give shards their final -of-N names, and write the index."""
        self.flush()
        count = len(self.shards)
        weight_map: dict[str, str] = {}
        for number, keys in enumerate(self.shards, 1):
            final = f"model-{number:05d}-of-{count:05d}.safetensors"
            (self.output_dir / f"model-{number:05d}.safetensors").rename(
                self.output_dir / final
            )
            weight_map.update(dict.fromkeys(keys, final))
        index = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": dict(sorted(weight_map.items())),
        }
        (self.output_dir / INDEX_FILENAME).write_text(
            json.dumps(index, indent=2) + "\n", encoding="utf-8"
        )
        return count


def _expert_keys(config: FishDrama3Config) -> set[str]:
    return {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"
        for layer in range(config.text.num_hidden_layers)
        for expert in range(config.text.num_experts)
        for proj in EXPERT_PROJECTIONS
        for kind in ("weight", "weight_scale_inv")
    }


def convert_fish_drama_3(
    input_dir: Path,
    output_dir: Path,
    *,
    quantization: ExpertQuantization,
    shard_bytes: int = SHARD_BYTES,
) -> Path:
    if quantization.mode != "affine":
        raise NotImplementedError("only affine expert quantization is supported")
    payload = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    config = FishDrama3Config.from_dict(payload)
    if not config.fp8_experts:
        raise ValueError(f"{input_dir} is not the upstream block-FP8 Drama 3 release")
    codec_pth = input_dir / "codec.pth"
    if not codec_pth.is_file():
        raise FileNotFoundError(f"missing {codec_pth}")

    index = ShardedCheckpointIndex.from_directory(input_dir)
    # mx.load is lazy, but a tensor stays resident while anything references it,
    # so each one leaves its shard dict as it is consumed: peak memory is one
    # layer's experts plus one output shard, not the whole checkpoint.
    shards = {name: mx.load(str(input_dir / name)) for name in index.shard_names}

    def tensor(key: str) -> mx.array:
        return shards[index.weight_map[key]].pop(key)

    expert_keys = {k for k in index.weight_map if UPSTREAM_EXPERT_KEY.fullmatch(k)}
    expected = _expert_keys(config)
    if expert_keys != expected:
        raise ValueError(
            f"routed experts: {len(expected - expert_keys)} missing, "
            f"{len(expert_keys - expected)} unexpected"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardWriter(output_dir, shard_bytes)
    for key in sorted(index.weight_map.keys() - expert_keys):
        name, value = to_mlx_dense(key, tensor(key))
        if value.dtype != mx.bfloat16:
            raise ValueError(f"{key}: expected bfloat16, got {value.dtype}")
        writer.add(name, value)

    experts = config.text.num_experts
    started = time.perf_counter()
    for layer in range(config.text.num_hidden_layers):
        base = f"model.language_model.layers.{layer}.mlp.experts"
        for proj in EXPERT_PROJECTIONS:
            weight = mx.stack(
                [tensor(f"{base}.{e}.{proj}.weight") for e in range(experts)]
            )
            scale = mx.stack(
                [tensor(f"{base}.{e}.{proj}.weight_scale_inv") for e in range(experts)]
            )
            # Quantizing bf16 with bf16 scales keeps the experts on MLX's native
            # bf16 gather_qmm path.
            dense = dequantize_fp8_blocks(weight, scale).astype(mx.bfloat16)
            packed, scales, biases = mx.quantize(
                dense,
                group_size=quantization.group_size,
                bits=quantization.bits,
                mode=quantization.mode,
            )
            mx.eval(packed, scales, biases)
            prefix = expert_prefix(layer, proj)
            writer.add(f"{prefix}.weight", packed)
            writer.add(f"{prefix}.scales", scales)
            writer.add(f"{prefix}.biases", biases)
            del weight, scale, dense
        mx.clear_cache()
        print(
            f"layer {layer + 1}/{config.text.num_hidden_layers} experts quantized "
            f"({time.perf_counter() - started:.0f}s, "
            f"peak {mx.get_peak_memory() / 2**30:.1f} GiB)",
            flush=True,
        )
    shard_count = writer.close()

    convert_codec_pth_to_assets(codec_pth, output_dir / "codec-mlx")
    converted = {k: v for k, v in payload.items() if k != "quantization_config"}
    converted["quantization"] = quantization.to_dict()
    (output_dir / "config.json").write_text(
        json.dumps(converted, indent=2) + "\n", encoding="utf-8"
    )
    for name in SUPPORTING_FILES:
        shutil.copy2(input_dir / name, output_dir / name)
    print(
        f"wrote {shard_count} shards ({writer.total_bytes / 1e9:.1f} GB) and "
        f"codec-mlx to {output_dir}"
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Local copy of fishaudio/drama-3.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expert-bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--group-size", type=int, default=64, choices=[32, 64, 128])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_fish_drama_3(
        args.input_dir,
        args.output_dir,
        quantization=ExpertQuantization(
            bits=args.expert_bits, group_size=args.group_size
        ),
    )


if __name__ == "__main__":
    main()
