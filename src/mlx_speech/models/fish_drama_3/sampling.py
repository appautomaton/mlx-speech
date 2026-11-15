"""Drama 3 sampling: Fish Audio's reference sampling (fish-speech #1368), released
with the weights of 2026-10-29.

A draw keeps rank r of the tempered candidates when r < top_k, cum[r] <= top_p and
p[r] >= epsilon_cutoff, and always keeps the most probable candidate; top_k < 1,
top_p >= 1 and epsilon_cutoff <= 0 turn their rule off. Only the main token has an
epsilon cutoff. A main draw that repeats one of the last REPEAT_WINDOW_FRAMES main
tokens, eos included, becomes the most probable candidate. Residual codes are drawn
only among the codes the codec's codebooks hold.
"""

from __future__ import annotations

import mlx.core as mx

MIN_TEMPERATURE = 1e-5
REPEAT_WINDOW_FRAMES = 16
MAIN_EPSILON_CUTOFF = 1e-3
# Codes per residual codebook of the codec; the codebook heads are 4096 wide.
RESIDUAL_CODEBOOK_SIZE = 1024


@mx.compile
def reference_candidates(
    logits: mx.array,
    temperature: float,
    top_k: int,
    top_p: float,
    epsilon_cutoff: float,
    legal_columns: int | None,
) -> tuple[mx.array, mx.array]:
    """Ids best first and their tempered logits, -inf where a rule drops them.

    Every rule reads the probabilities over all columns; legal_columns then drops
    the later ids, and the most probable legal id stays whatever the rules say.
    """
    scaled = logits.astype(mx.float32) / max(temperature, MIN_TEMPERATURE)
    ids = mx.argsort(-scaled, axis=-1)
    values = mx.take_along_axis(scaled, ids, axis=-1)
    probabilities = mx.softmax(values, axis=-1)
    ranks = mx.arange(values.shape[-1])
    kept = probabilities >= epsilon_cutoff
    if top_k >= 1:
        kept = kept & (ranks < top_k)
    if top_p < 1.0:
        kept = kept & (mx.cumsum(probabilities, axis=-1) <= top_p)
    if legal_columns is None:
        best = ranks == 0
    else:
        legal = ids < legal_columns
        kept = kept & legal
        best = ranks == mx.argmax(legal.astype(mx.int32), axis=-1, keepdims=True)
    return ids, mx.where(kept | best, values, -mx.inf)


def reference_sample(
    logits: mx.array,
    temperature: float,
    top_k: int,
    top_p: float,
    *,
    epsilon_cutoff: float = 0.0,
    legal_columns: int | None = None,
    keys: mx.array | None = None,
) -> mx.array:
    """One id per row of logits [rows, columns]; keys [rows] give each row its own
    draw, independent of the rows beside it. None draws from the global generator."""
    ids, values = reference_candidates(
        logits, temperature, top_k, top_p, epsilon_cutoff, legal_columns
    )
    if keys is None:
        choice = mx.random.categorical(values, axis=-1)
    else:
        choice = mx.vmap(lambda row, key: mx.random.categorical(row, key=key))(
            values, keys
        )
    return mx.take_along_axis(ids, choice[..., None], axis=-1)[..., 0]


def replace_repeats(first: mx.array, best: mx.array, history: mx.array) -> mx.array:
    """best where the first draw is among the last REPEAT_WINDOW_FRAMES main tokens
    of its row's history [..., frames]."""
    repeated = (history[..., -REPEAT_WINDOW_FRAMES:] == first[..., None]).any(axis=-1)
    return mx.where(repeated, best, first)
