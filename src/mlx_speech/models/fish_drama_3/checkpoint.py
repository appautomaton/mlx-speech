"""Drama 3 checkpoint layout: upstream key names and block-FP8 expert decoding.

Upstream stores each routed expert as three float8_e4m3fn matrices, one fp32 scale
per 128x128 block, under ``model.language_model.layers.L.mlp.experts.E.*``. The MLX
checkpoint stacks a layer's experts per projection as ``[experts, out, in]`` under
``backbone.layers.L.mlp.experts.{gate_proj,up_proj,down_proj}``.
"""

from __future__ import annotations

import re

import mlx.core as mx

from .config import FP8_BLOCK_SIZE

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
