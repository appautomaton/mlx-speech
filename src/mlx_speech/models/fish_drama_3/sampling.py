"""Drama 3 sampling: top-k / top-p as upstream, plus repetition-aware resampling."""

from __future__ import annotations

import mlx.core as mx

MIN_TEMPERATURE = 1e-5
# Repetition-aware sampling of the main token, fixed by the training recipe: a
# semantic draw that repeats one of the last RAS_WINDOW_FRAMES main tokens is
# replaced by a second draw with these parameters.
RAS_WINDOW_FRAMES = 16
RAS_TEMPERATURE = 1.0
RAS_TOP_K = 30
RAS_TOP_P = 0.95


def top_k_top_p_candidates(
    logits: mx.array, temperature: float, top_k: int, top_p: float
) -> tuple[mx.array, mx.array]:
    """Ids and tempered logits of the tokens a draw may pick, best first.

    Matches upstream exactly: rank by logit, drop ranks >= top_k and every token
    whose cumulative probability (over the full vocabulary) exceeds top_p, but
    always keep the best one. Only the top_k best are sorted; the full softmax
    normalizer comes from one logsumexp. Dropped entries hold -inf.
    """
    scaled = logits.astype(mx.float32) / max(temperature, MIN_TEMPERATURE)
    k = min(top_k, scaled.shape[-1])
    ids = mx.argpartition(-scaled, kth=k - 1, axis=-1)[..., :k]
    values = mx.take_along_axis(scaled, ids, axis=-1)
    order = mx.argsort(-values, axis=-1)
    ids = mx.take_along_axis(ids, order, axis=-1)
    values = mx.take_along_axis(values, order, axis=-1)
    normalizer = mx.logsumexp(scaled, axis=-1, keepdims=True)
    cumulative = mx.cumsum(mx.exp(values - normalizer), axis=-1)
    dropped = cumulative > top_p
    dropped[..., 0] = False
    return ids, mx.where(dropped, -mx.inf, values)


def sample_top_k_top_p(
    logits: mx.array, temperature: float, top_k: int, top_p: float
) -> mx.array:
    """One token id per row of logits [..., vocab]."""
    ids, values = top_k_top_p_candidates(logits, temperature, top_k, top_p)
    choice = mx.random.categorical(values, axis=-1)
    return mx.take_along_axis(ids, choice[..., None], axis=-1)[..., 0]


def resample_repeats(
    first: mx.array, retry: mx.array, history: mx.array, is_semantic: mx.array
) -> mx.array:
    """The retry draw where the first draw is a semantic token found in history.

    Prompt rows in the window hold the user turn's eos, and a stop must never be
    redrawn, so only semantic draws are replaced.
    """
    repeated = (history[-RAS_WINDOW_FRAMES:] == first[..., None]).any(axis=-1)
    return mx.where(repeated & is_semantic, retry, first)
