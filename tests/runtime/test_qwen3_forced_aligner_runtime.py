"""Qwen3-ForcedAligner on real speech: timestamps against measured silences.

Pauses are measured from the waveform itself (frame RMS more than 35 dB below
the loud frames), so the gates do not depend on hand-labelled times. The model
predicts 80 ms classes; boundary tolerances are two classes.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import mlx_speech
from mlx_speech.audio import load_audio
from mlx_speech.generation.qwen3_forced_aligner import Qwen3ForcedAligner


ROOT = Path("models/qwen3_forced_aligner_0_6b")
PACKAGES = ("mlx-bf16", "mlx-int8")
# Tracked in git: four TTS speakers with long pauses between turns.
ENGLISH_CLIP = Path("examples/audio/vibevoice_4speaker_convo.wav")
ENGLISH_TEXT = (
    "We need to set a lunch before the next meeting starts. I want ramen, but only if the "
    "line is short today. Tacos are faster, and the truck is already parked outside. I "
    "brought salad, so I am mainly here to judge the debate. The ramen place was packed "
    "yesterday around noon. Then tacos sound safer, and I can still get something spicy. "
    "Good tacos, it is, and I am ordering extra chips for the table. Perfect. I will walk "
    "with you and keep eating my extremely responsible salad."
)
# Local only (gitignored): opens with about 0.5 s of silence.
CHINESE_CLIP = Path("models/netease/confucius4_r2t2/samples/test.wav")
CHINESE_TEXT = "之前有顾客自己带酒水，也没加收钱或者不让喝。"
TOLERANCE_S = 0.16

pytestmark = pytest.mark.runtime


def _aligner(package: str):
    model_dir = ROOT / package
    if not (model_dir / "model.safetensors").is_file():
        pytest.skip(f"Qwen3-ForcedAligner {package} package not present: {model_dir}")
    return mlx_speech.align.load(model_dir)


@pytest.fixture(scope="module", params=PACKAGES)
def aligner(request):  # type: ignore[no-untyped-def]
    return _aligner(request.param)


def _waveform(path: Path) -> np.ndarray:
    if not path.is_file():
        pytest.skip(f"Speech clip not present: {path}")
    samples, _ = load_audio(path, sample_rate=16000, mono=True)
    return np.array(samples, dtype=np.float32)


def _silences(waveform: np.ndarray, *, min_s: float) -> list[tuple[float, float]]:
    hop, win = 160, 400
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(waveform, win // 2), win)[::hop]
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=-1)) + 1e-9)
    quiet = db < np.percentile(db, 99) - 35.0
    runs, start = [], None
    for index, flag in enumerate([*quiet, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if (index - start) * hop / 16000 >= min_s:
                runs.append((start * hop / 16000, index * hop / 16000))
            start = None
    return runs


def _assert_well_formed(items, units: list[str], duration_s: float) -> None:
    assert [item.text for item in items] == units
    times = [t for item in items for t in (item.start_s, item.end_s)]
    assert times == sorted(times)
    assert 0.0 <= times[0] and times[-1] <= duration_s


def _assert_pauses_align(items, waveform: np.ndarray, *, min_s: float) -> int:
    """Every inner pause of at least ``min_s`` sits between two units, edge to edge."""

    pauses = [
        (a, b)
        for a, b in _silences(waveform, min_s=min_s)
        if items[0].start_s < a and b < items[-1].end_s
    ]
    for a, b in pauses:
        before = max(i for i, item in enumerate(items) if item.end_s <= b)
        assert abs(items[before].end_s - a) <= TOLERANCE_S, (items[before], a)
        assert abs(items[before + 1].start_s - b) <= TOLERANCE_S, (items[before + 1], b)
    return len(pauses)


def test_english_turn_pauses_fall_between_words(aligner):
    waveform = _waveform(ENGLISH_CLIP)
    items = aligner.generate(waveform, ENGLISH_TEXT, language="en")

    words = [w.strip(".,") for w in ENGLISH_TEXT.split()]
    _assert_well_formed(items, words, waveform.shape[0] / 16000)
    assert _assert_pauses_align(items, waveform, min_s=0.5) >= 6
    leading = _silences(waveform, min_s=0.3)[0]
    assert leading[0] == 0.0 and abs(items[0].start_s - leading[1]) <= TOLERANCE_S


def test_chinese_characters_skip_the_leading_silence(aligner):
    waveform = _waveform(CHINESE_CLIP)
    items = aligner.generate(str(CHINESE_CLIP), CHINESE_TEXT, language="zh")

    _assert_well_formed(items, list("之前有顾客自己带酒水也没加收钱或者不让喝"), waveform.shape[0] / 16000)
    leading = _silences(waveform, min_s=0.3)[0]
    assert leading[0] == 0.0 and abs(items[0].start_s - leading[1]) <= TOLERANCE_S
    assert _assert_pauses_align(items, waveform, min_s=0.5) == 1  # the comma


def test_int8_matches_bf16_within_one_class():
    waveform = _waveform(ENGLISH_CLIP)
    bf16 = _aligner("mlx-bf16").generate(waveform, ENGLISH_TEXT, language="en")
    int8 = _aligner("mlx-int8").generate(waveform, ENGLISH_TEXT, language="en")

    diffs = np.abs(np.array([i[1:] for i in bf16]) - np.array([i[1:] for i in int8]))
    assert diffs.max() <= 0.08 + 1e-9
    assert np.mean(diffs == 0) >= 0.95


def test_time_head_matches_a_float64_oracle():
    model_dir = ROOT / "mlx-int8"
    if not (model_dir / "model.safetensors").is_file():
        pytest.skip(f"Qwen3-ForcedAligner mlx-int8 package not present: {model_dir}")
    runtime = Qwen3ForcedAligner.from_dir(model_dir)
    inputs = runtime.prepare(_waveform(ENGLISH_CLIP), ENGLISH_TEXT, language="en")
    hidden = runtime.timestamp_hidden_states(inputs)
    logits = np.array(runtime.model.text_decoder.project_logits(hidden), dtype=np.float64)

    weight = np.array(runtime.model.text_decoder.lm_head.weight.astype(mx.float32), dtype=np.float64)
    oracle = np.array(hidden.astype(mx.float32), dtype=np.float64) @ weight.T
    np.testing.assert_allclose(logits, oracle, rtol=0, atol=1e-3)
    assert np.array_equal(logits.argmax(axis=-1), oracle.argmax(axis=-1))
