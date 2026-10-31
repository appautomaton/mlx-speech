"""Hard gates for Audio8-ASR-Infinite live streaming and rolling context."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlx_speech.asr._adapters.audio8_asr_infinite import Audio8ASRAdapter
from mlx_speech.audio import load_audio
from mlx_speech.models.audio8_asr_infinite.model import Audio8ASRModel
from mlx_speech.models.audio8_asr_infinite.text_decoder import text_cache_length

CHECKPOINTS = (
    Path("models/edge0/audio8_asr_infinite/mlx-bf16"),
    Path("models/edge0/audio8_asr_infinite/mlx-int8"),
)
CLIP = Path(".references/mlx-audio/mlx_audio/stt/tests/mega_asr/fixtures/clean.wav")
RAGGED_SIZES = (1, 137, 4001, 16_000)
ROLLING_CONTEXT = 64
ROLLING_TRIM_INTERVAL_MS = 800
TILES = 8

pytestmark = pytest.mark.skipif(
    not all((path / "model.safetensors").is_file() for path in CHECKPOINTS)
    or not CLIP.is_file(),
    reason="Audio8 bf16/int8 checkpoints or pinned speech fixture not present",
)


@pytest.fixture(scope="module", params=CHECKPOINTS, ids=("bf16", "int8"))
def runtime(request) -> tuple[Audio8ASRModel, np.ndarray]:  # type: ignore[no-untyped-def]
    waveform, sample_rate = load_audio(CLIP, sample_rate=16_000, mono=True)
    assert sample_rate == 16_000
    samples = np.asarray(waveform, dtype=np.float32).reshape(-1)
    return Audio8ASRModel.from_dir(request.param), samples


def _run(model: Audio8ASRModel, samples: np.ndarray, *, sizes=None, **session_kwargs):  # type: ignore[no-untyped-def]
    session = model.stream_session(language="en", **session_kwargs)
    if sizes is None:
        session.feed(samples)
    else:
        offset = index = 0
        while offset < samples.shape[0]:
            size = sizes[index % len(sizes)]
            session.feed(samples[offset : offset + size])
            offset += size
            index += 1
    session.finalize()
    return session


def test_ragged_feeds_equal_one_feed(
    runtime: tuple[Audio8ASRModel, np.ndarray],
) -> None:
    model, samples = runtime
    whole = _run(model, samples)
    ragged = _run(model, samples, sizes=RAGGED_SIZES)

    assert whole.generated_token_ids
    assert ragged.generated_token_ids == whole.generated_token_ids


def test_rolling_window_is_inert_until_it_fills(
    runtime: tuple[Audio8ASRModel, np.ndarray],
) -> None:
    model, samples = runtime
    rolling = _run(model, samples)
    unbounded = _run(model, samples, rolling_context_tokens=0)

    assert rolling.generated_token_ids == unbounded.generated_token_ids


def test_rolling_window_bounds_both_caches_and_keeps_transcribing(
    runtime: tuple[Audio8ASRModel, np.ndarray],
) -> None:
    model, samples = runtime
    tiled = np.tile(samples, TILES)
    frame_seconds = model.config.audio_tower_frame_ms / 1000
    native_frames = model.config.audio_config.max_position_embeddings
    assert tiled.shape[0] / 16_000 / frame_seconds > native_frames, (
        "fixture too short to force an audio-cache rebase"
    )

    single = len(_run(model, samples).result().tokens)
    session = _run(
        model,
        tiled,
        rolling_context_tokens=ROLLING_CONTEXT,
        rolling_trim_interval_ms=ROLLING_TRIM_INTERVAL_MS,
    )

    assert text_cache_length(session.text_cache) <= ROLLING_CONTEXT
    assert session.audio_cache.position_offset < native_frames
    # Collapse to silence or a repetition loop falls outside this band.
    visible = len(session.result().tokens)
    assert 0.6 * TILES * single <= visible <= 1.5 * TILES * single


def test_adapter_matches_runtime(
    runtime: tuple[Audio8ASRModel, np.ndarray],
) -> None:
    model, samples = runtime
    output = Audio8ASRAdapter(model).generate(CLIP, language="en")

    assert output.language == "en"
    assert output.text == model.transcribe(samples, language="en").text
