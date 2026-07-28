"""Hard gates for live Nemotron waveform and encoder streaming."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.audio import load_audio
from mlx_speech.models.nemotron_asr.model import NemotronASRModel
from mlx_speech.models.nemotron_asr.streaming import (
    StreamingEncoder,
    StreamingMelFrontend,
)

CHECKPOINT = Path("models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16")
CLIP = Path(".references/mlx-audio/mlx_audio/stt/tests/mega_asr/fixtures/clean.wav")
CONTEXT = (56, 3)
RAGGED_SIZES = (1, 137, 4001, 16_000)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file() or not CLIP.is_file(),
    reason="Nemotron checkpoint or pinned speech fixture not present",
)


@pytest.fixture(scope="module")
def runtime() -> tuple[NemotronASRModel, mx.array]:
    waveform, sample_rate = load_audio(CLIP, sample_rate=16_000, mono=True)
    assert sample_rate == 16_000
    return NemotronASRModel.from_dir(CHECKPOINT), waveform


def _ragged(waveform: mx.array):  # type: ignore[no-untyped-def]
    offset = 0
    index = 0
    while offset < waveform.shape[0]:
        size = RAGGED_SIZES[index % len(RAGGED_SIZES)]
        yield waveform[offset : offset + size]
        offset += size
        index += 1


def test_streamed_encoder_frames_equal_offline_at_native_chunk_size(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, waveform = runtime
    features, lengths = model.preprocessor(waveform)
    offline, offline_lengths = model.encoder(
        features,
        lengths,
        CONTEXT,
    )

    mel_stream = StreamingMelFrontend(model.preprocessor, model.config.preprocessor)
    encoder_stream = StreamingEncoder(model.encoder, att_context_size=CONTEXT)
    cache_ids = tuple(id(buffer) for buffer in encoder_stream.cache_buffers)
    assert all(
        layer.attention.capacity == 56
        for layer in encoder_stream.layers
    )
    assert all(
        layer.convolution.capacity == 8
        for layer in encoder_stream.layers
    )
    chunks = []
    for pcm in _ragged(waveform):
        assert pcm.shape[0] > 0
        mel = mel_stream.feed(pcm)
        assert mel_stream.residual_sample_count < model.config.preprocessor.hop_length
        chunks.extend(encoder_stream.feed(mel))
    chunks.extend(encoder_stream.feed(mel_stream.finalize(), final=True))
    streamed = mx.concatenate(chunks, axis=1)
    mx.eval(offline, offline_lengths, streamed)

    assert tuple(id(buffer) for buffer in encoder_stream.cache_buffers) == cache_ids
    assert streamed.shape == offline.shape
    assert streamed.shape[1] == int(offline_lengths[0].item())
    np.testing.assert_allclose(
        np.asarray(streamed.astype(mx.float32)),
        np.asarray(offline.astype(mx.float32)),
        rtol=1e-4,
        atol=1.5e-4,
    )


def test_ragged_live_tokens_equal_offline_and_finalize_flushes_tail(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, waveform = runtime
    offline = model.transcribe(
        waveform,
        language="en-US",
        att_context_size=CONTEXT,
    )
    session = model.stream_session(
        language="en-US",
        att_context_size=CONTEXT,
    )
    cache_ids = tuple(id(buffer) for buffer in session.encoder.cache_buffers)
    streamed_tokens = []
    for pcm in _ragged(waveform):
        streamed_tokens.extend(session.feed(pcm))
        assert session.mel.residual_sample_count < 160
    tail = session.finalize()
    streamed_tokens.extend(tail)

    assert tail
    assert tuple(streamed_tokens) == offline.tokens
    assert session.result().tokens == offline.tokens
    assert session.result().text == offline.text
    assert session.predictor_state is not None
    assert session.last_token == offline.tokens[-1]
    assert tuple(id(buffer) for buffer in session.encoder.cache_buffers) == cache_ids


def test_one_feed_and_many_ragged_feeds_are_identical(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, waveform = runtime
    single = model.stream_session(language="en-US", att_context_size=CONTEXT)
    single_tokens = (*single.feed(waveform), *single.finalize())

    ragged = model.stream_session(language="en-US", att_context_size=CONTEXT)
    ragged_tokens = []
    for pcm in _ragged(waveform):
        ragged_tokens.extend(ragged.feed(pcm))
    ragged_tokens.extend(ragged.finalize())

    assert tuple(ragged_tokens) == single_tokens
    assert ragged.result().frame_indices == single.result().frame_indices


def test_predictor_state_survives_a_sub_hop_feed_boundary(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, waveform = runtime
    offline = model.transcribe(waveform, language="en-US", att_context_size=CONTEXT)
    session = model.stream_session(language="en-US", att_context_size=CONTEXT)
    emitted = []
    offset = 0
    while not emitted:
        next_offset = min(offset + 16_000, waveform.shape[0])
        emitted.extend(session.feed(waveform[offset:next_offset]))
        offset = next_offset
        assert offset < waveform.shape[0], "fixture should emit before its final sample"

    hidden_before = np.asarray(session.predictor_state[0]).copy()
    cell_before = np.asarray(session.predictor_state[1]).copy()
    last_token = session.last_token
    assert session.feed(waveform[offset : offset + 1]) == ()
    offset += 1

    np.testing.assert_array_equal(
        np.asarray(session.predictor_state[0]), hidden_before
    )
    np.testing.assert_array_equal(np.asarray(session.predictor_state[1]), cell_before)
    assert session.last_token == last_token

    emitted.extend(session.feed(waveform[offset:]))
    emitted.extend(session.finalize())
    assert session.result().tokens == offline.tokens


def test_finalize_is_idempotent_and_feed_after_finalize_raises(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, _ = runtime
    session = model.stream_session(language="en-US", att_context_size=CONTEXT)
    session.feed(mx.zeros((137,), dtype=mx.float32))

    session.finalize()

    assert session.finalize() == ()
    with pytest.raises(RuntimeError, match="finalized"):
        session.feed(mx.zeros((1,), dtype=mx.float32))
