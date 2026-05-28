"""Binary `attention_mask` → additive log-space mask.

This is the conversion the upstream code applies right before the connector
runs:

    additive = (mask.int() - 1).to(dtype).reshape(B, 1, -1, T) * finfo(dtype).max

For valid tokens (mask=1) the additive offset is ``0``. For padding (mask=0)
it's ``-finfo(dtype).max`` — saturating the softmax to zero without using
literal ``-inf`` (which can propagate NaN if a row is entirely masked).

Reference: `.references/DramaBox/ltx2/ltx_core/text_encoders/gemma/embeddings_processor.py:15-19`
"""

from __future__ import annotations

import mlx.core as mx


def convert_to_additive_mask(attention_mask: mx.array, dtype: mx.Dtype) -> mx.array:
    """Convert ``[B, T]`` binary mask to ``[B, 1, 1, T]`` additive bias.

    Args:
        attention_mask: int/bool mask, ``1`` for valid, ``0`` for pad.
        dtype: target compute dtype for the additive bias.

    Returns:
        Additive bias of shape ``[B, 1, 1, T]`` in ``dtype``:
        ``0`` at valid slots, ``-finfo(dtype).max`` at padded slots.
    """
    B, T = attention_mask.shape
    mask_minus_one = attention_mask.astype(mx.int32) - 1  # 0 → -1, 1 → 0
    mask_minus_one = mask_minus_one.astype(dtype).reshape(B, 1, 1, T)
    return mask_minus_one * mx.array(mx.finfo(dtype).max, dtype=dtype)


__all__ = ["convert_to_additive_mask"]
