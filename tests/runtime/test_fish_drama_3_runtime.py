"""Real-weight Drama 3 runtime checks: encode_codes, batched history, and audio infilling."""

import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.audio import write_wav
from mlx_speech.generation.fish_drama_3 import Drama3Chunk, FishDrama3Runtime
from mlx_speech.models.fish_drama_3.prompt import (
    FIM_PREFIX,
    FIM_SUFFIX,
    build_infill_prompt,
)

MODEL_DIR = Path("models/fish_drama_3/base")
FRAME_LENGTH = 2048


@pytest.fixture(scope="module")
def runtime() -> FishDrama3Runtime:
    if not (MODEL_DIR / "config.json").is_file():
        pytest.skip("No local Drama 3 release")
    rt = FishDrama3Runtime.from_dir(MODEL_DIR, dense_int8=True)
    yield rt
    del rt
    gc.collect()
    mx.clear_cache()


@pytest.fixture(scope="module")
def base_chunk(runtime: FishDrama3Runtime) -> Drama3Chunk:
    text = "In the nineteen fifties, Soviet bankers held dollars abroad."
    chunks = runtime.synthesize_batch(
        [text], seeds=[0], duration_seconds=3.5, max_new_tokens=160
    )
    chunk = chunks[0]
    assert chunk.codes.shape[0] == 10
    assert chunk.codes.shape[1] >= 40
    return chunk


@pytest.mark.runtime
def test_real_tokenizer_resolves_fim_tokens(runtime: FishDrama3Runtime):
    prefix = np.ones((10, 2), dtype=np.int32)
    suffix = np.full((10, 2), 2, dtype=np.int32)
    prefix_id = runtime.tokenizer.token_to_id(FIM_PREFIX)
    suffix_id = runtime.tokenizer.token_to_id(FIM_SUFFIX)

    rows = build_infill_prompt("Hello.", prefix, suffix, runtime.tokenizer, runtime.config)

    assert prefix_id is not None and prefix_id in rows[:, 0].tolist()
    assert suffix_id is not None and suffix_id in rows[:, 0].tolist()


@pytest.mark.runtime
def test_encode_codes_encodes_wav_to_ten_codebooks(
    runtime: FishDrama3Runtime, base_chunk: Drama3Chunk, tmp_path: Path
):
    wav_path = tmp_path / "base.wav"
    write_wav(wav_path, base_chunk.waveform, sample_rate=runtime.sample_rate)

    codes = runtime.encode_codes(wav_path)

    assert codes.dtype == np.int32
    assert codes.shape[0] == 10
    assert codes.shape[1] > 0


@pytest.mark.runtime
def test_synthesize_batch_with_history_produces_finite_audio(
    runtime: FishDrama3Runtime, base_chunk: Drama3Chunk
):
    lines = runtime.synthesize_batch(
        [
            "They moved the deposits to London.",
            "No freeze could touch them there.",
        ],
        seeds=[1, 2],
        history=[(base_chunk.text, base_chunk.codes)],
        duration_seconds=[2.0, 2.0],
    )

    assert len(lines) == 2
    for chunk in lines:
        assert chunk.codes.shape[0] == 10
        assert chunk.codes.shape[1] > 0
        assert chunk.waveform.ndim == 1
        assert chunk.waveform.shape[0] == chunk.codes.shape[1] * FRAME_LENGTH
        assert mx.all(mx.isfinite(chunk.waveform)).item()


@pytest.mark.runtime
def test_infill_preserves_prefix_and_suffix_and_decodes_full_waveform(
    runtime: FishDrama3Runtime, base_chunk: Drama3Chunk
):
    prefix = base_chunk.codes[:, :16]
    suffix = base_chunk.codes[:, -16:]

    chunk = runtime.infill(
        "In the nineteen fifties, European bankers held dollars abroad.",
        prefix,
        suffix,
        duration_seconds=0.8,
        seed=0,
    )

    assert chunk.codes.dtype == np.int32
    assert chunk.codes.shape[0] == 10
    assert chunk.codes.shape[1] > prefix.shape[1] + suffix.shape[1]
    np.testing.assert_array_equal(chunk.codes[:, : prefix.shape[1]], prefix)
    np.testing.assert_array_equal(chunk.codes[:, -suffix.shape[1] :], suffix)
    assert chunk.waveform.ndim == 1
    assert chunk.waveform.shape[0] == chunk.codes.shape[1] * FRAME_LENGTH
    assert mx.all(mx.isfinite(chunk.waveform)).item()
