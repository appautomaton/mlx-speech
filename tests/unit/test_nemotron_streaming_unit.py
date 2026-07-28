from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_speech.models.nemotron_asr.config import ConformerArgs, PreprocessArgs
from mlx_speech.models.nemotron_asr.encoder import FastConformerEncoder
from mlx_speech.models.nemotron_asr.feature_extraction import NemotronPreprocessor
from mlx_speech.models.nemotron_asr.streaming import (
    FixedFrameCache,
    StreamingEncoder,
    StreamingMelFrontend,
)

_FEATURE_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "nemotron" / "features.npz"
)


def _small_encoder() -> FastConformerEncoder:
    return FastConformerEncoder(
        ConformerArgs(
            feat_in=16,
            n_layers=2,
            d_model=16,
            n_heads=4,
            ff_expansion_factor=2,
            subsampling_conv_channels=4,
            att_context_size=((8, 1),),
            default_att_context_size=(8, 1),
            pos_emb_max_len=64,
        )
    )


def test_fixed_cache_preserves_ring_order_without_reallocation() -> None:
    cache = FixedFrameCache(4, 1, dtype=mx.float32)
    buffer_id = id(cache.buffer)

    cache.append(mx.array([[[1.0], [2.0], [3.0]]]))
    cache.append(mx.array([[[4.0], [5.0]]]))
    values = cache.values()
    mx.eval(values)

    assert id(cache.buffer) == buffer_id
    np.testing.assert_array_equal(
        np.asarray(values).reshape(-1), np.asarray([2.0, 3.0, 4.0, 5.0])
    )


def test_streaming_mel_matches_offline_across_sub_hop_feeds() -> None:
    with np.load(_FEATURE_FIXTURE) as fixture:
        waveform = fixture["waveform"]
    preprocessor = NemotronPreprocessor(PreprocessArgs())
    offline, offline_lengths = preprocessor(waveform)
    stream = StreamingMelFrontend(preprocessor, preprocessor.args)

    chunks = []
    offset = 0
    sizes = (1, 137, 4001, 53, 160)
    index = 0
    while offset < waveform.shape[0]:
        size = sizes[index % len(sizes)]
        chunks.append(stream.feed(waveform[offset : offset + size]))
        offset += size
        index += 1
    assert stream.residual_sample_count == waveform.shape[0] % 160
    chunks.append(stream.finalize())
    streamed = mx.concatenate([chunk for chunk in chunks if chunk.shape[1]], axis=1)
    mx.eval(offline, offline_lengths, streamed)

    valid = int(offline_lengths[0].item())
    assert streamed.shape == (1, valid, 128)
    np.testing.assert_allclose(
        np.asarray(streamed), np.asarray(offline[:, :valid]), rtol=3e-4, atol=3e-4
    )
    assert stream.buffered_sample_count <= preprocessor.args.n_fft


def test_streamed_encoder_is_identical_to_offline_native_chunks() -> None:
    encoder = _small_encoder()
    mel = mx.sin(mx.arange(65 * 16, dtype=mx.float32) * 0.017).reshape(1, 65, 16)
    lengths = mx.array([65], dtype=mx.int32)
    offline, offline_lengths = encoder(mel, lengths, (8, 1))
    stream = StreamingEncoder(encoder, att_context_size=(8, 1))
    cache_ids = tuple(id(buffer) for buffer in stream.cache_buffers)

    outputs = []
    offset = 0
    for size in (1, 7, 13, 3, 19, 22):
        outputs.extend(stream.feed(mel[:, offset : offset + size]))
        offset += size
    outputs.extend(stream.feed(mel[:, offset:], final=True))
    streamed = mx.concatenate(outputs, axis=1)
    mx.eval(offline, offline_lengths, streamed)

    assert tuple(id(buffer) for buffer in stream.cache_buffers) == cache_ids
    assert streamed.shape == offline.shape == (1, 9, 16)
    np.testing.assert_allclose(
        np.asarray(streamed), np.asarray(offline), rtol=2e-5, atol=2e-5
    )


def test_finalize_emits_subsampling_tail_after_exact_mel_chunk() -> None:
    encoder = _small_encoder()
    stream = StreamingEncoder(encoder, att_context_size=(8, 1))
    mel = mx.zeros((1, 16, 16))

    before_finalize = stream.feed(mel)
    tail = stream.feed(mx.zeros((1, 0, 16)), final=True)

    assert sum(chunk.shape[1] for chunk in before_finalize) == 2
    assert sum(chunk.shape[1] for chunk in tail) == 1
