"""Convert a multimodal Gemma 3 12B IT checkpoint into a text-only MLX 4-bit backbone.

Pure-MLX conversion: no torch, no mlx-lm, no transformers. Reads the source
``safetensors`` shards via ``mx.load`` (which handles bf16 natively), drops the
vision tower / multimodal projector, strips the ``language_model.model.`` prefix
so the layout is flat (matching the existing ``ltx_accel_gemma3_text`` artifact
under ``agentcubic/models/ltx-2.3/gemma-3-12b-pt-4bit-text``), affine-quantizes
every linear weight with ``mx.quantize`` (group_size=64, bits=4), and writes a
sharded MLX-format checkpoint plus a slim ``config.json``.

Norm weights (``*_layernorm.weight``, ``*_norm.weight``, final ``norm.weight``)
are kept as bf16 — they're 1-D and tiny, and quantizing them hurts quality
without saving meaningful memory.

Usage::

    python scripts/convert/gemma_3_text_4bit.py \
        --hf-path models/gemma_3_12b_it_backbone/original \
        --out-dir models/gemma_3_12b_it_backbone/mlx-4bit \
        --bits 4 --group-size 64
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import mlx.core as mx


# Keys whose ``.weight`` is a small 1-D norm/embedding scale and stays bf16.
_NORM_SUFFIXES = (
    "_layernorm.weight",
    "_norm.weight",
)
# Top-level prefixes to drop entirely (vision branch is unused for text encoding).
_DROP_PREFIXES = ("vision_tower.", "multi_modal_projector.")
# Prefix to strip so saved keys match the flat ``embed_tokens.* / layers.*``
# convention used by ``ltx_accel_gemma3_text``.
_KEEP_PREFIX = "language_model.model."


def _is_norm(key: str) -> bool:
    return key.endswith(_NORM_SUFFIXES)


def _is_droppable(key: str) -> bool:
    return any(key.startswith(p) for p in _DROP_PREFIXES)


def _flat_key(key: str) -> str | None:
    """Return the flat-layout key for a source key, or ``None`` to drop it."""
    if _is_droppable(key):
        return None
    if not key.startswith(_KEEP_PREFIX):
        # Unknown top-level — refuse to guess.
        return None
    return key[len(_KEEP_PREFIX):]


def _load_source_shards(hf_dir: Path) -> dict[str, mx.array]:
    shards = sorted(hf_dir.glob("model-*-of-*.safetensors"))
    if not shards:
        # Fall back to single-file checkpoint.
        shards = sorted(hf_dir.glob("model.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors shards under {hf_dir}")
    weights: dict[str, mx.array] = {}
    for shard in shards:
        # ``mx.load`` returns mlx arrays with the on-disk dtype (bf16 preserved).
        weights.update(mx.load(str(shard)))
    return weights


def _quantize_one(w: mx.array, bits: int, group_size: int) -> tuple[mx.array, mx.array, mx.array]:
    """Quantize a single weight tensor and force-eval it (so we don't OOM on
    a giant unevaluated graph at save time)."""
    q, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    mx.eval(q, scales, biases)
    return q, scales, biases


def _split_into_shards(items: list[tuple[str, mx.array]], max_bytes: int) -> list[dict[str, mx.array]]:
    """Greedy fixed-size sharding. Each shard accumulates up to ``max_bytes``."""
    shards: list[dict[str, mx.array]] = [{}]
    current = 0
    for key, arr in items:
        nbytes = arr.nbytes
        if current and current + nbytes > max_bytes:
            shards.append({})
            current = 0
        shards[-1][key] = arr
        current += nbytes
    return shards


def _write_index(out_dir: Path, shard_files: list[str], shard_weights: list[dict[str, mx.array]]) -> None:
    total = sum(sum(a.nbytes for a in s.values()) for s in shard_weights)
    weight_map: dict[str, str] = {}
    for fname, arrays in zip(shard_files, shard_weights):
        for key in arrays:
            weight_map[key] = fname
    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


def _write_config(
    src_config_path: Path,
    out_config_path: Path,
    bits: int,
    group_size: int,
) -> None:
    src = json.loads(src_config_path.read_text())
    text_cfg = src.get("text_config", {})
    cfg = {
        "architectures": ["Gemma3Model"],
        "model_type": "gemma3_text_backbone",
        "name_or_path": out_config_path.parent.name,
        "source_model_name_or_path": src.get("name_or_path", "google/gemma-3-12b-it-qat-q4_0-unquantized"),
        "source_model_type": src.get("model_type", "gemma3"),
        "source_architectures": src.get("architectures", []),
        "library_name": "mlx",
        "format": "mlx",
        "backbone_only": True,
        "license": "gemma",
        "torch_dtype": src.get("torch_dtype", "bfloat16"),
        "text_config": text_cfg,
        "quantization": {"group_size": group_size, "bits": bits, "mode": "affine"},
        "quantization_config": {"group_size": group_size, "bits": bits, "mode": "affine"},
        "eos_token_id": src.get("eos_token_id", text_cfg.get("eos_token_id")),
    }
    out_config_path.write_text(json.dumps(cfg, indent=4))


def _copy_tokenizer(src_dir: Path, out_dir: Path) -> None:
    """Copy tokenizer artefacts verbatim. Keeps ``tokenizer.json``,
    ``tokenizer.model``, special-token maps, and chat templates."""
    keep = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "chat_template.json",
        "chat_template.jinja",
        "generation_config.json",
    }
    for name in keep:
        src = src_dir / name
        if src.is_file():
            shutil.copy2(src, out_dir / name)


def convert(hf_path: Path, out_dir: Path, bits: int, group_size: int, shard_bytes: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading source shards from {hf_path}")
    src = _load_source_shards(hf_path)

    # Remap + drop. Track which keys went where for sanity printing.
    kept_keys: dict[str, mx.array] = {}
    dropped = 0
    for k, v in src.items():
        new_k = _flat_key(k)
        if new_k is None:
            dropped += 1
            continue
        kept_keys[new_k] = v
    print(f"  source keys: {len(src)} | kept: {len(kept_keys)} | dropped: {dropped}")

    # Quantize linears, keep norms as-is.
    out_arrays: dict[str, mx.array] = {}
    n_quant = 0
    n_plain = 0
    for k in sorted(kept_keys):
        w = kept_keys[k]
        if _is_norm(k) or w.ndim < 2:
            # Norms and any pure 1-D scalars: keep at bf16.
            out_arrays[k] = w
            n_plain += 1
            continue
        # Linear / embedding matrix: affine-quantize.
        q, scales, biases = _quantize_one(w, bits=bits, group_size=group_size)
        out_arrays[f"{k}"] = q
        out_arrays[f"{k.removesuffix('.weight')}.scales" if k.endswith(".weight") else f"{k}.scales"] = scales
        out_arrays[f"{k.removesuffix('.weight')}.biases" if k.endswith(".weight") else f"{k}.biases"] = biases
        n_quant += 1
    print(f"  quantized weights: {n_quant} | unquantized: {n_plain} | total output keys: {len(out_arrays)}")

    # Stable-sort keys so shard layout is reproducible across runs.
    items = sorted(out_arrays.items())
    shards = _split_into_shards(items, max_bytes=shard_bytes)
    n_shards = len(shards)
    shard_files: list[str] = []
    for idx, arrays in enumerate(shards, start=1):
        fname = f"model-{idx:05d}-of-{n_shards:05d}.safetensors"
        path = out_dir / fname
        mx.save_safetensors(str(path), arrays, metadata={"format": "mlx"})
        size_gb = sum(a.nbytes for a in arrays.values()) / 1e9
        print(f"  wrote {fname}: {len(arrays)} keys, {size_gb:.2f} GB")
        shard_files.append(fname)

    _write_index(out_dir, shard_files, shards)
    _write_config(hf_path / "config.json", out_dir / "config.json", bits=bits, group_size=group_size)
    _copy_tokenizer(hf_path, out_dir)
    print(f"done. output at {out_dir}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--hf-path", type=Path, required=True, help="Source HF Gemma 3 dir (multimodal).")
    p.add_argument("--out-dir", type=Path, required=True, help="Destination MLX 4-bit dir.")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group-size", type=int, default=64)
    p.add_argument("--shard-bytes", type=int, default=5_300_000_000,
                   help="Max bytes per output shard (default ~5.3 GB, matches existing artifact).")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    convert(
        hf_path=args.hf_path,
        out_dir=args.out_dir,
        bits=args.bits,
        group_size=args.group_size,
        shard_bytes=args.shard_bytes,
    )


if __name__ == "__main__":
    main()
