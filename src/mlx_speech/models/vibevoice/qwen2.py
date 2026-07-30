"""Compatibility exports for the shared Qwen2 decoder trunk."""

import mlx.core as mx

from .._qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2KVCache,
    Qwen2LayerCache,
    Qwen2MLP,
    Qwen2Model,
    Qwen2Output,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
)

VIBEVOICE_ACTIVATION_DTYPE = mx.bfloat16

__all__ = [
    "Qwen2Attention",
    "Qwen2DecoderLayer",
    "Qwen2KVCache",
    "Qwen2LayerCache",
    "Qwen2MLP",
    "Qwen2Model",
    "Qwen2Output",
    "Qwen2RMSNorm",
    "Qwen2RotaryEmbedding",
    "VIBEVOICE_ACTIVATION_DTYPE",
]
