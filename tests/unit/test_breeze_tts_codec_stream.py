"""Chunked codec decode matches one offline decode of the same codes."""

from __future__ import annotations

import mlx.core as mx

from mlx_speech.models.breeze_tts.codec import (
    CodecDecodeStream,
    CodecDecoder,
    DecoderBlock,
    PreTransformer,
    UpsampleConv,
    _upsample_step,
    _WindowKV,
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


def test_codec_attention_sees_only_the_official_72_frame_window() -> None:
    mx.random.seed(3)
    hidden = mx.random.normal((1, 100, 1024))

    # One layer: frame 71 still sees frame 0; frame 72 and later do not.
    # Stacked layers widen that reach by 71 frames each, as upstream does.
    single = PreTransformer()
    single.layers = single.layers[:1]
    before = single(hidden)
    after = single(hidden.at[:, 0, :].add(10.0))
    assert float(mx.max(mx.abs(after[:, 71] - before[:, 71])).item()) > 0.0
    assert float(mx.max(mx.abs(after[:, 72:] - before[:, 72:])).item()) == 0.0

    # The 0.01 layer-scale init mutes attention; full scale exposes window bugs.
    # An off-by-one window moves these outputs by about 5e-2. Single-query
    # attention rounds differently from the masked kernel by up to about 4e-3.
    transformer = PreTransformer()
    for layer in transformer.layers:
        layer.self_attn_layer_scale.scale = mx.ones((512,))
        layer.mlp_layer_scale.scale = mx.ones((512,))
    offline = transformer(hidden)
    for splits in ([100], [4] * 25, [70, 1, 1, 1, 27], [50, 30, 20]):
        caches = [_WindowKV() for _ in transformer.layers]
        parts, start = [], 0
        for size in splits:
            parts.append(
                transformer.step(hidden[:, start : start + size], caches, start)
            )
            start += size
        streamed = mx.concatenate(parts, axis=1)
        assert float(mx.max(mx.abs(streamed - offline)).item()) < 1e-2, splits
        assert all(int(cache.keys.shape[2]) == 71 for cache in caches)


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
