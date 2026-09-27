"""Pure-helper tests for the R2T2 streaming baseline harness."""

from __future__ import annotations

import importlib.util
from collections.abc import Sequence
from pathlib import Path
import sys

import numpy as np
import pytest

from mlx_speech.models.confucius4_r2t2.metrics import R2T2StreamTrace
from mlx_speech.models.confucius4_r2t2.streaming import R2T2StreamSession

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "eval"
    / "benchmark_confucius4_r2t2_streaming.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location("benchmark_r2t2_streaming_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _StubDecoder:
    def __call__(self, prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
        return "hello"


def _tokenize(text: str) -> list[int]:
    return [ord(char) for char in text]


def _detokenize(ids: Sequence[int]) -> str:
    return "".join(chr(int(token)) for token in ids)


def test_window_plan_follows_the_first_window_lookahead() -> None:
    module = _load_script_module()

    plan, tail = module.window_plan(9_000, chunk_samples=3200, lookahead_samples=3200)
    assert plan == [(0, 6400)]
    assert tail == (6400, 9000)

    plan, tail = module.window_plan(12_800, chunk_samples=3200, lookahead_samples=3200)
    assert plan == [(0, 6400), (6400, 9600), (9600, 12800)]
    assert tail is None

    plan, tail = module.window_plan(0, chunk_samples=3200, lookahead_samples=3200)
    assert plan == []
    assert tail is None


def test_window_plan_matches_the_session_schedule() -> None:
    module = _load_script_module()

    for total in (6_400, 9_000, 12_800):
        plan, tail = module.window_plan(total, chunk_samples=3200, lookahead_samples=3200)
        trace = R2T2StreamTrace()
        session = R2T2StreamSession(
            _StubDecoder(),
            _tokenize,
            _detokenize,
            language="English",
            chunk_ms=200,
            lookahead_ms=200,
            unfixed_chunk_num=3,
            unfixed_token_num=0,
            trace=trace,
        )
        audio = np.zeros(total, dtype=np.float32)
        for start, end in plan:
            session.feed(audio[start:end])
        if tail is not None:
            session.feed(audio[tail[0]:tail[1]])
        session.finalize()

        expected = list(plan) + ([tail] if tail is not None else [])
        assert [(step.sample_start, step.sample_end) for step in trace.steps] == expected


def test_arrival_seconds_use_the_planned_endpoints() -> None:
    module = _load_script_module()

    arrivals = module.arrival_seconds(9_000, chunk_samples=3200, lookahead_samples=3200)
    assert arrivals == [6400 / 16_000, 9000 / 16_000]


def test_assert_schedule_accepts_a_match_and_rejects_a_drift() -> None:
    module = _load_script_module()

    records = [
        {"sample_start": 0, "sample_end": 6400},
        {"sample_start": 6400, "sample_end": 9000},
    ]
    module.assert_schedule(
        records,
        [
            {"sample_start": 0, "sample_end": 6400},
            {"sample_start": 6400, "sample_end": 9000},
        ],
    )
    with pytest.raises(AssertionError):
        module.assert_schedule(
            records,
            [
                {"sample_start": 0, "sample_end": 6400},
                {"sample_start": 6400, "sample_end": 8000},
            ],
        )
    with pytest.raises(AssertionError):
        module.assert_schedule(records, [{"sample_start": 0, "sample_end": 6400}])


def test_build_concatenated_audio_round_robins_and_truncates() -> None:
    module = _load_script_module()
    clips = [
        np.full(3, 1.0, dtype=np.float32),
        np.full(2, 2.0, dtype=np.float32),
    ]

    assert module.build_concatenated_audio(clips, target_samples=5).tolist() == [
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
    ]
    assert module.build_concatenated_audio(clips, target_samples=8).tolist() == [
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        1.0,
        1.0,
        1.0,
    ]

    with pytest.raises(ValueError):
        module.build_concatenated_audio([], target_samples=10)
    with pytest.raises(ValueError):
        module.build_concatenated_audio(clips, target_samples=0)
    with pytest.raises(ValueError):
        module.build_concatenated_audio([np.zeros(0, dtype=np.float32)], target_samples=10)


def test_percentile_interpolates_and_rejects_empty() -> None:
    module = _load_script_module()

    assert module.percentile([1.0], 0.95) == pytest.approx(1.0)
    assert module.percentile([0.0, 1.0], 0.5) == pytest.approx(0.5)
    assert module.percentile([0.0, 1.0, 2.0], 0.5) == pytest.approx(1.0)
    assert module.percentile([0.0, 1.0, 2.0, 3.0], 0.95) == pytest.approx(2.8500000000000005)
    with pytest.raises(ValueError):
        module.percentile([], 0.5)


def test_summarize_reports_spread_and_handles_empty() -> None:
    module = _load_script_module()

    assert module.summarize([]) == {
        "count": 0,
        "p50_seconds": None,
        "p95_seconds": None,
        "max_seconds": None,
        "mean_seconds": None,
        "spread_seconds": None,
    }

    stats = module.summarize([0.10, 0.20, 0.30, 0.40])
    assert stats["count"] == 4
    assert stats["p50_seconds"] == pytest.approx(0.25)
    assert stats["max_seconds"] == pytest.approx(0.40)
    assert stats["mean_seconds"] == pytest.approx(0.25)
    assert stats["spread_seconds"] == pytest.approx(0.30)


def test_median_pass_picks_the_middle_processing_ratio() -> None:
    module = _load_script_module()

    runs = [
        {"processing_audio_ratio": 1.4},
        {"processing_audio_ratio": 1.0},
        {"processing_audio_ratio": 1.2},
        {"processing_audio_ratio": 1.2},
    ]
    assert module.median_pass(runs)["processing_audio_ratio"] == pytest.approx(1.2)
    with pytest.raises(ValueError):
        module.median_pass([])


def test_classify_needs_latency_and_backlog_both_inside_the_target() -> None:
    module = _load_script_module()
    target = 0.16

    good = {"queue": {"step_p95_seconds": 0.15, "max_queue_wait_seconds": 0.05}}
    slow = {"queue": {"step_p95_seconds": 0.20, "max_queue_wait_seconds": 0.05}}
    behind = {"queue": {"step_p95_seconds": 0.15, "max_queue_wait_seconds": 0.20}}
    unknown = {"queue": {"step_p95_seconds": None, "max_queue_wait_seconds": 0.0}}

    assert module.classify(good, step_target_seconds=target)["target_met"] is True
    assert module.classify(slow, step_target_seconds=target)["target_met"] is False
    assert module.classify(slow, step_target_seconds=target)["p95_over_target"] is True
    assert module.classify(behind, step_target_seconds=target)["target_met"] is False
    assert module.classify(behind, step_target_seconds=target)["backlog_over_target"] is True
    assert module.classify(unknown, step_target_seconds=target)["target_met"] is False


def test_ms_to_samples_matches_the_default_schedule() -> None:
    module = _load_script_module()

    assert module.ms_to_samples(160) == 2560
    assert module.ms_to_samples(80) == 1280
    assert module.ms_to_samples(0) == 0
