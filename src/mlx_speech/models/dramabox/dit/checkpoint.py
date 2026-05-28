"""Checkpoint loader for the DramaBox DiT.

All DiT keys live under the ``model.diffusion_model.`` prefix. We strip that
and feed the remainder to `model.load_weights(...)`. The audio attention,
FFN, and AdaLN modules use Python-list children, so the saved keys
(`transformer_blocks.5.audio_attn1.to_out.0.weight`, etc.) map directly.

There is no Conv2d in the DiT, so we do not need the encoder-decoder
permute step here.
"""

from __future__ import annotations

import mlx.core as mx

from .model import LTXModel

_PREFIX = "model.diffusion_model."


def load_dit_weights(model: LTXModel, state: dict[str, mx.array]) -> int:
    """Load ``model.diffusion_model.*`` keys into the DiT.

    Skips the connector (`audio_embeddings_connector.*`) and the aggregate
    projection (`text_embedding_projection.*`) which belong to the prompt
    pipeline (Stage 2b), not the DiT itself.
    """
    sub: dict[str, mx.array] = {}
    for k, v in state.items():
        if not k.startswith(_PREFIX):
            continue
        tail = k[len(_PREFIX):]
        # Skip prompt-pipeline keys that share this prefix in the audio
        # components shard but belong elsewhere.
        if tail.startswith("audio_embeddings_connector"):
            continue
        sub[tail] = v
    if not sub:
        raise KeyError(f"No DiT keys under prefix {_PREFIX!r}")
    model.load_weights(list(sub.items()), strict=True)
    return len(sub)


__all__ = ["load_dit_weights"]
