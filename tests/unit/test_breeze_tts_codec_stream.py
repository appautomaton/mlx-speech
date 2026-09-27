"""Chunked codec decode matches one offline decode of the same codes."""

from __future__ import annotations

import mlx.core as mx

from mlx_speech.models.breeze_tts.codec import (
    CodecDecodeStream,
    CodecDecoder,
    DecoderBlock,
    UpsampleConv,
    _upsample_step,
)


def _decode_chunks(
    decoder: CodecDecoder, codes: mx.array, splits: list[int]
) -> mx.array:
    stream = CodecDecodeStream(decoder)
    start = 0
    parts = []
    for size in splits:
        parts.append(stream.push(codes[:, :, start : start + size]))
        start += size
    return mx.concatenate(parts, axis=-1)


def _nonzero_codec(decoder: CodecDecoder) -> None:
    """Random init has zero codebooks and zero biases, which hide stream bugs."""

    for quantizer in (decoder.quantizer.rvq_first, decoder.quantizer.rvq_rest):
        for layer in quantizer.vq.layers:
            codebook = layer.codebook
            codebook.embedding_sum = mx.random.normal(codebook.embedding_sum.shape)
    for layer in decoder.decoder:
        if not isinstance(layer, DecoderBlock):
            continue
        for part in layer.block:
            if isinstance(part, UpsampleConv) and part.conv.bias is not None:
                part.conv.bias = mx.random.normal(part.conv.bias.shape)


def test_streaming_codec_matches_offline_decode_across_chunk_boundaries() -> None:
    mx.random.seed(0)
    decoder = CodecDecoder()
    _nonzero_codec(decoder)
    codes = mx.random.randint(0, 2048, (1, 16, 7))
    offline = decoder(codes)
    streamed = _decode_chunks(decoder, codes, [1, 1, 1, 1, 1, 1, 1])
    regrouped = _decode_chunks(decoder, codes, [3, 3, 1])

    assert streamed.shape == offline.shape
    assert regrouped.shape == offline.shape
    difference = mx.max(mx.abs(offline - streamed))
    mx.eval(difference)
    # Later Snake activations amplify overlap-add rounding, so this is looser
    # than the direct transposed-convolution check below.
    assert float(difference.item()) < 1e-3
    difference = mx.max(mx.abs(offline - regrouped))
    mx.eval(difference)
    assert float(difference.item()) < 1e-3


def test_a_new_codec_stream_does_not_keep_the_previous_utterance() -> None:
    mx.random.seed(1)
    decoder = CodecDecoder()
    _nonzero_codec(decoder)
    codes = mx.random.randint(0, 2048, (1, 16, 4))
    first = CodecDecodeStream(decoder)
    first.push(codes[:, :, :2])
    fresh = CodecDecodeStream(decoder).push(codes)
    offline = decoder(codes)
    difference = mx.max(mx.abs(offline - fresh))
    mx.eval(difference)
    assert float(difference.item()) < 1e-4


def test_upsample_overlap_includes_bias_once() -> None:
    mx.random.seed(2)
    layer = UpsampleConv(8, 4, 8)
    layer.conv.bias = mx.random.normal(layer.conv.bias.shape)
    hidden = mx.random.normal((1, 6, 8))
    offline = layer(hidden)
    tail = None
    parts = []
    for start in (0, 2, 4):
        chunk, tail = _upsample_step(layer, hidden[:, start : start + 2], tail)
        parts.append(chunk)
    difference = mx.max(mx.abs(offline - mx.concatenate(parts, axis=1)))
    mx.eval(difference)
    assert float(difference.item()) < 1e-5
