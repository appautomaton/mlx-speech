"""Runtime checks for the local Confucius4-R2T2 bf16 package.

Weights and sample audio live under ``models/`` (gitignored), so every test
here skips when they are absent. The streaming test forces ``language`` on
purpose: ``samples/test.wav`` opens with roughly 320 ms of silence, and with
``language=None`` the first window's answer is ``language None``. The published
state machine treats a tagless answer as "nothing to commit", so the
auto-language path never grows on this clip. That is upstream behavior, not a
port defect, so the gate drives the forced-language path instead.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import mlx_speech.asr as asr
from mlx_speech.audio import load_audio


MODEL_DIR = Path("models/netease/confucius4_r2t2/mlx-bf16")
SAMPLE = Path("models/netease/confucius4_r2t2/samples/test.wav")
QWEN3_ASR_DIR = Path("models/qwen3_asr_1_7b/mlx-int8")

# Robust fragments of the published transcript; they also fail on the
# degenerate repetition loop a broken tied lm_head produces.
PHRASE_HEAD = "之前有顾客自己带酒水"
PHRASE_TAIL = "不让喝"

FEED_SAMPLES = 1600

pytestmark = pytest.mark.runtime


def _require_local_assets() -> None:
    missing = [
        str(path)
        for path in (MODEL_DIR / "config.json", MODEL_DIR / "model.safetensors", SAMPLE)
        if not path.exists()
    ]
    if missing:
        pytest.skip(
            "Confucius4-R2T2 local runtime assets not present: " + ", ".join(missing)
        )


def test_r2t2_offline_generate_returns_the_transcript():
    _require_local_assets()
    model = asr.load(str(MODEL_DIR))

    result = model.generate(str(SAMPLE))

    assert result.language == "Chinese"
    assert len(result.text) >= 15
    assert PHRASE_HEAD in result.text
    assert PHRASE_TAIL in result.text


def test_r2t2_streaming_commits_monotonically_and_finalize_matches():
    _require_local_assets()
    model = asr.load(str(MODEL_DIR))
    waveform, sample_rate = load_audio(str(SAMPLE), sample_rate=16000, mono=True)
    assert sample_rate == 16000
    pcm = np.asarray(waveform, dtype=np.float32)

    session = model.stream_session(
        language="Chinese",
        chunk_ms=160,
        lookahead_ms=160,
    )

    committed = ""
    for start in range(0, pcm.shape[0], FEED_SAMPLES):
        update = session.feed(pcm[start : start + FEED_SAMPLES])
        assert update.committed.startswith(committed)
        committed = update.committed

    result = session.finalize()

    # Finish releases the token unfixed_token_num was holding back. For this
    # clip that token is the final ideographic full stop.
    assert result.text.startswith(committed)
    assert result.text.removeprefix(committed) == "。"
    assert result.language == "Chinese"
    assert len(committed) >= 15
    assert PHRASE_HEAD in committed
    assert PHRASE_TAIL in committed


def test_r2t2_marker_is_required_for_the_r2t2_family():
    from mlx_speech.asr._registry import _resolve_asr_family

    if not (MODEL_DIR / "config.json").exists():
        pytest.skip(f"{MODEL_DIR} missing")

    assert _resolve_asr_family(MODEL_DIR) == "r2t2"
    if (QWEN3_ASR_DIR / "config.json").exists():
        assert _resolve_asr_family(QWEN3_ASR_DIR) == "qwen3"
