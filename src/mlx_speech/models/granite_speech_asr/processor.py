"""Granite Speech prompt/input embedding helpers."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def mask_audio_token_ids(
    input_ids: mx.array,
    *,
    audio_token_id: int,
    replacement_id: int = 0,
) -> mx.array:
    """Replace audio sentinel IDs before passing IDs to the text embedding table."""
    return mx.where(input_ids == audio_token_id, replacement_id, input_ids)


def replace_audio_embeddings(
    input_ids: mx.array,
    token_embeddings: mx.array,
    audio_features: mx.array,
    *,
    audio_token_id: int,
) -> mx.array:
    """Substitute projected audio features at `<|audio|>` token positions."""
    ids_np = np.array(input_ids)
    squeeze_ids = False
    if ids_np.ndim == 1:
        ids_np = ids_np[None, :]
        squeeze_ids = token_embeddings.ndim == 2
    if ids_np.ndim != 2:
        raise ValueError(f"input_ids must have shape [T] or [B, T], got {input_ids.shape}")

    embeds_np = np.array(token_embeddings.astype(mx.float32))
    if embeds_np.ndim == 2:
        embeds_np = embeds_np[None, :, :]
    audio_np = np.array(audio_features.astype(mx.float32))
    if audio_np.ndim != 3:
        raise ValueError(f"audio_features must have shape [B, A, D], got {audio_features.shape}")
    if embeds_np.ndim != 3:
        raise ValueError(f"token_embeddings must have shape [B, T, D], got {token_embeddings.shape}")
    if ids_np.shape[:2] != embeds_np.shape[:2]:
        raise ValueError(
            f"input_ids shape {ids_np.shape} does not match token_embeddings shape {embeds_np.shape[:2]}"
        )
    if ids_np.shape[0] != audio_np.shape[0]:
        raise ValueError(
            f"audio batch {audio_np.shape[0]} does not match input batch {ids_np.shape[0]}"
        )
    if embeds_np.shape[2] != audio_np.shape[2]:
        raise ValueError(
            f"audio hidden size {audio_np.shape[2]} does not match token hidden size {embeds_np.shape[2]}"
        )

    for batch_idx in range(ids_np.shape[0]):
        positions = np.where(ids_np[batch_idx] == audio_token_id)[0]
        if len(positions) != audio_np.shape[1]:
            raise ValueError(
                f"audio token count mismatch for batch {batch_idx}: "
                f"{len(positions)} prompt token(s), {audio_np.shape[1]} audio feature(s)"
            )
        embeds_np[batch_idx, positions] = audio_np[batch_idx]

    result = mx.array(embeds_np).astype(token_embeddings.dtype)
    if squeeze_ids:
        return result[0]
    return result
