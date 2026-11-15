"""Drama 3 checkpoint layout: upstream key names and block-FP8 expert decoding.

Upstream stores each routed expert as three float8_e4m3fn matrices, one fp32 scale
per 128x128 block, under ``model.language_model.layers.L.mlp.experts.E.*``. The MLX
model stacks a layer's experts per projection as ``[experts, out, in]`` under
``backbone.layers.L.mlp.experts.{gate_proj,up_proj,down_proj}``. MLX has no kernel for
block FP8, so the experts are requantized in memory to MLX affine int8 as the release
is loaded.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import mlx.core as mx

from ...checkpoints.sharded import ShardedCheckpointIndex
from .config import FP8_BLOCK_SIZE, ExpertQuantization, FishDrama3Config

EXPERT_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
UPSTREAM_EXPERT_KEY = re.compile(
    r"model\.language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight|weight_scale_inv)"
)
# Upstream prefix -> MLX prefix, for every tensor but the routed experts.
_PREFIX_MAP = (
    ("model.language_model.", "backbone."),
    ("lm_head.", "backbone.lm_head."),
    ("audio_decoder.codebook_embeddings.", "audio_decoder.codebook_embeddings."),
    ("audio_decoder_mtp_shared.layers.", "audio_decoder.shared_layers."),
    ("audio_decoder_mtp.layers.", "audio_decoder.residual_layers."),
    ("audio_decoder_mtp.heads.", "audio_decoder.codebook_heads."),
)
_CONV1D_SUFFIX = ".linear_attn.conv1d.weight"
# float8_e4m3fn has no infinities; S.1111.111 is NaN. A NaN weight would poison
# its whole quantization group, and MLX before 0.32.3 decoded it as +-480.
_FP8_NAN_BITS = 0x7F


# How the routed experts are held when the upstream release is loaded directly.
UPSTREAM_EXPERT_QUANTIZATION = ExpertQuantization(bits=8, group_size=64)


def expert_prefix(layer: int, projection: str) -> str:
    return f"backbone.layers.{layer}.mlp.experts.{projection}"


def remap_key(key: str) -> str:
    """MLX name of an upstream tensor other than a routed expert."""
    if UPSTREAM_EXPERT_KEY.fullmatch(key):
        raise ValueError(f"{key} is a routed expert; stack it with its layer")
    for old, new in _PREFIX_MAP:
        if key.startswith(old):
            return new + key[len(old) :]
    raise ValueError(f"unknown Drama 3 checkpoint key {key!r}")


def to_mlx_dense(key: str, value: mx.array) -> tuple[str, mx.array]:
    """MLX name and layout of an upstream non-expert tensor."""
    name = remap_key(key)
    if name.endswith(_CONV1D_SUFFIX):
        # Depthwise conv: torch [channels, 1, kernel] -> MLX [channels, kernel, 1].
        if value.ndim != 3 or value.shape[1] != 1:
            raise ValueError(
                f"{key}: expected [channels, 1, kernel], got {value.shape}"
            )
        value = value.transpose(0, 2, 1)
    return name, value


def dequantize_fp8_blocks(weight: mx.array, scale_inv: mx.array) -> mx.array:
    """Block FP8 ``[..., rows, cols]`` (uint8 bits) with fp32 ``[..., rows/128,
    cols/128]`` scales -> float32 ``[..., rows, cols]``.

    Each product is rounded once to fp32, a relative error of at most 2**-24.
    """
    if weight.dtype != mx.uint8:
        raise TypeError(f"expected float8_e4m3fn bits as uint8, got {weight.dtype}")
    *lead, rows, cols = weight.shape
    if rows % FP8_BLOCK_SIZE or cols % FP8_BLOCK_SIZE:
        raise ValueError(
            f"FP8 weight {tuple(weight.shape)} is not tiled by {FP8_BLOCK_SIZE}"
        )
    row_blocks, col_blocks = rows // FP8_BLOCK_SIZE, cols // FP8_BLOCK_SIZE
    if tuple(scale_inv.shape) != (*lead, row_blocks, col_blocks):
        raise ValueError(
            f"scale shape {tuple(scale_inv.shape)} does not tile weight "
            f"{tuple(weight.shape)} by {FP8_BLOCK_SIZE}"
        )
    if mx.any((weight & _FP8_NAN_BITS) == _FP8_NAN_BITS).item():
        raise ValueError("FP8 weight holds NaN codes")
    values = mx.from_fp8(weight, dtype=mx.float32).reshape(
        *lead, row_blocks, FP8_BLOCK_SIZE, col_blocks, FP8_BLOCK_SIZE
    )
    scales = scale_inv.astype(mx.float32)[..., :, None, :, None]
    return (values * scales).reshape(*lead, rows, cols)


def quantize_min_max(
    w: mx.array, group_size: int, bits: int
) -> tuple[mx.array, mx.array, mx.array]:
    """MLX affine quantization ``(weight, scales, biases)`` whose levels span each
    group from its minimum to its maximum.

    The layout is mx.quantize's, so mx.dequantize and gather_qmm read it as is, but
    the range is chosen differently: on Drama 3's experts this one leaves about a
    fifth less error against the FP8 values. Scales and biases are bf16, which keeps
    the experts on MLX's native bf16 gather_qmm path.
    """
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be 2, 4 or 8; got {bits}")
    *lead, rows, cols = w.shape
    if cols % group_size:
        raise ValueError(f"{cols} columns are not tiled by groups of {group_size}")
    levels = 2**bits - 1
    groups = w.astype(mx.float32).reshape(*lead, rows, cols // group_size, group_size)
    low = groups.min(axis=-1, keepdims=True)
    high = groups.max(axis=-1, keepdims=True)
    # A constant group gets a scale too small to matter and is restored as its bias.
    scales = (mx.maximum(high - low, 1e-30) / levels).astype(mx.bfloat16)
    biases = low.astype(mx.bfloat16)
    q = mx.round((groups - biases.astype(mx.float32)) / scales.astype(mx.float32))
    q = mx.clip(q, 0, levels).astype(mx.uint32)
    per_word = 32 // bits
    q = q.reshape(*lead, rows, cols // per_word, per_word)
    packed = q[..., 0]
    for slot in range(1, per_word):
        packed = packed | (q[..., slot] << (bits * slot))
    return packed, scales[..., 0], biases[..., 0]


def requantize_experts(
    weight: mx.array, scale_inv: mx.array, quantization: ExpertQuantization
) -> tuple[mx.array, mx.array, mx.array]:
    """Stacked block-FP8 experts ``[experts, out, in]`` as MLX affine-quantized
    (weight, scales, biases), fitted to the exact FP8 values in fp32: rounding them
    to bf16 first would add error and hold a second copy."""
    if quantization.mode != "affine":
        raise NotImplementedError("only affine expert quantization is supported")
    exact = dequantize_fp8_blocks(weight, scale_inv)
    return quantize_min_max(exact, quantization.group_size, quantization.bits)


def upstream_weights(
    model_dir: Path,
    config: FishDrama3Config,
    quantization: ExpertQuantization,
) -> Iterator[tuple[str, mx.array]]:
    """MLX-named weights of the upstream release in model_dir.

    Every tensor but the routed experts comes as stored, lazily. The experts are
    requantized one projection of one layer at a time and evaluated before the next
    is read, so the FP8 source of each is released as soon as it is consumed: the
    extra memory is one projection's worth, not a second copy of the model.
    """
    if quantization.mode != "affine":
        raise NotImplementedError("only affine expert quantization is supported")
    index = ShardedCheckpointIndex.from_directory(model_dir)
    expert_keys = {k for k in index.weight_map if UPSTREAM_EXPERT_KEY.fullmatch(k)}
    expected = {
        f"model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}"
        for layer in range(config.text.num_hidden_layers)
        for expert in range(config.text.num_experts)
        for proj in EXPERT_PROJECTIONS
        for kind in ("weight", "weight_scale_inv")
    }
    if expert_keys != expected:
        raise ValueError(
            f"routed experts: {len(expected - expert_keys)} missing, "
            f"{len(expert_keys - expected)} unexpected"
        )
    # mx.load is lazy, but a tensor stays resident while anything references it,
    # so each one leaves its shard dict as it is consumed.
    shards = {name: mx.load(str(model_dir / name)) for name in index.shard_names}

    def tensor(key: str) -> mx.array:
        return shards[index.weight_map[key]].pop(key)

    for key in sorted(index.weight_map.keys() - expert_keys):
        name, value = to_mlx_dense(key, tensor(key))
        if value.dtype != mx.bfloat16:
            raise ValueError(f"{key}: expected bfloat16, got {value.dtype}")
        yield name, value

    experts = config.text.num_experts
    for layer in range(config.text.num_hidden_layers):
        base = f"model.language_model.layers.{layer}.mlp.experts"
        for proj in EXPERT_PROJECTIONS:
            weight = mx.stack(
                [tensor(f"{base}.{e}.{proj}.weight") for e in range(experts)]
            )
            scale = mx.stack(
                [tensor(f"{base}.{e}.{proj}.weight_scale_inv") for e in range(experts)]
            )
            packed, scales, biases = requantize_experts(weight, scale, quantization)
            mx.eval(packed, scales, biases)
            del weight, scale
            prefix = expert_prefix(layer, proj)
            yield f"{prefix}.weight", packed
            yield f"{prefix}.scales", scales
            yield f"{prefix}.biases", biases
