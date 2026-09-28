"""Streamed Breeze frames use the same codes as the offline waveform."""

from __future__ import annotations

import mlx.core as mx

from mlx_speech.generation.breeze_tts import generate_breeze, generate_breeze_stream
from mlx_speech.models.breeze_tts.codec import CodecDecoder


class _Codec:
    def __init__(self) -> None:
        self.decoder = CodecDecoder()

    def decode(self, codes: mx.array) -> mx.array:
        return self.decoder(codes)


def _frames() -> list[mx.array]:
    mx.random.seed(0)
    return [mx.random.randint(0, 2048, (16,)) for _ in range(7)]


def test_streamed_chunks_match_one_offline_decode(monkeypatch) -> None:
    frames = _frames()

    def same_frames(*args, **kwargs):
        del args, kwargs
        yield from frames

    monkeypatch.setattr(
        "mlx_speech.generation.breeze_tts._iter_code_frames", same_frames
    )
    codec = _Codec()
    offline = generate_breeze(object(), codec, object(), "hello")
    streamed = mx.concatenate(
        list(generate_breeze_stream(object(), codec, object(), "hello", chunk_frames=3))
    )
    assert streamed.shape == offline.samples.shape
    difference = mx.max(mx.abs(offline.samples - streamed))
    mx.eval(difference)
    assert float(difference.item()) < 1e-4


def test_closing_a_partial_stream_does_not_change_the_next_request(monkeypatch) -> None:
    frames = _frames()

    def same_frames(*args, **kwargs):
        del args, kwargs
        yield from frames

    monkeypatch.setattr(
        "mlx_speech.generation.breeze_tts._iter_code_frames", same_frames
    )
    codec = _Codec()
    partial = generate_breeze_stream(object(), codec, object(), "hello", chunk_frames=3)
    next(partial)
    partial.close()
    streamed = mx.concatenate(
        list(generate_breeze_stream(object(), codec, object(), "hello", chunk_frames=3))
    )
    offline = generate_breeze(object(), codec, object(), "hello")
    difference = mx.max(mx.abs(offline.samples - streamed))
    mx.eval(difference)
    assert float(difference.item()) < 1e-4
