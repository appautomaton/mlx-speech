"""Rolling KV-cache operations for Audio8 streaming."""

from __future__ import annotations

import mlx.core as mx

from .audio_encoder import Audio8AudioCache
from .text_decoder import Audio8TextCache, text_cache_length


def apply_rope_delta(keys: mx.array, inv_freq: mx.array, delta: int) -> mx.array:
    if delta == 0 or keys.size == 0:
        return keys
    frequencies = inv_freq.astype(mx.float32) * float(delta)
    angles = mx.concatenate([frequencies, frequencies], axis=-1)
    shape = (1,) * (keys.ndim - 1) + (angles.shape[-1],)
    cos = mx.cos(angles).reshape(shape).astype(keys.dtype)
    sin = mx.sin(angles).reshape(shape).astype(keys.dtype)
    return keys * cos + _rotate_half(keys) * sin


def trim_text_cache(
    cache: Audio8TextCache,
    *,
    stable_prefix: int,
    max_tokens: int,
    inv_freq: mx.array,
) -> int:
    current_length = text_cache_length(cache)
    if stable_prefix < 0 or max_tokens <= 0 or stable_prefix > max_tokens:
        raise ValueError("Audio8 text cache target must include the stable prefix")
    trim_count = current_length - max_tokens
    if trim_count <= 0:
        return 0
    if stable_prefix + trim_count > current_length:
        raise ValueError("Audio8 text cache trim range is invalid")
    for layer in cache:
        keys, values = layer.fetch()
        head_keys = keys[:, :stable_prefix]
        head_values = values[:, :stable_prefix]
        tail_start = stable_prefix + trim_count
        tail_keys = apply_rope_delta(keys[:, tail_start:], inv_freq, -trim_count)
        tail_values = values[:, tail_start:]
        kept_keys = mx.concatenate([head_keys, tail_keys], axis=1)
        kept_values = mx.concatenate([head_values, tail_values], axis=1)
        next_keys = mx.zeros_like(layer.keys)
        next_values = mx.zeros_like(layer.values)
        next_keys[:, :max_tokens] = kept_keys
        next_values[:, :max_tokens] = kept_values
        layer.keys = next_keys
        layer.values = next_values
        layer.offset = max_tokens
    return trim_count


def rebase_audio_cache(
    cache: Audio8AudioCache,
    *,
    target_position: int,
    inv_freq: mx.array,
) -> int:
    trim_count = cache.position_offset - target_position
    if trim_count <= 0:
        return 0
    for layer in cache.layers:
        if layer.keys is None:
            continue
        keys = layer.keys[:, :, -target_position:]
        values = layer.values[:, :, -target_position:]
        positions = layer.positions[-target_position:]
        layer.replace(
            apply_rope_delta(keys, inv_freq, -trim_count),
            values,
            positions - trim_count,
        )
    cache.position_offset = target_position
    return trim_count


def _rotate_half(values: mx.array) -> mx.array:
    half = values.shape[-1] // 2
    return mx.concatenate([-values[..., half:], values[..., :half]], axis=-1)
