"""Unit tests for R2T2 streaming instrumentation and the queue simulation."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from mlx_speech.models.confucius4_r2t2.metrics import (
    PCM_BYTES_COPIED,
    PCM_SAMPLES_COPIED,
    R2T2StepMetrics,
    R2T2StreamTrace,
    simulate_stream_queue,
    stage,
)
from mlx_speech.models.confucius4_r2t2.streaming import R2T2StreamSession

_CHUNK_SAMPLES = 2560


def _tokenize(text: str) -> list[int]:
    return [ord(char) for char in text]


def _detokenize(ids: Sequence[int]) -> str:
    return "".join(chr(int(token)) for token in ids)


class _Script:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls: list[tuple[str, int, int]] = []

    def __call__(self, prefix: str, audio: np.ndarray, max_new_tokens: int) -> str:
        self.calls.append((prefix, int(audio.shape[0]), int(max_new_tokens)))
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index]


def _session(script, **kwargs) -> R2T2StreamSession:
    kwargs.setdefault("language", "English")
    kwargs.setdefault("lookahead_ms", 0)
    # Keep the injected prefix empty and the held-back token at zero so each
    # scripted output is exactly the committed text.
    kwargs.setdefault("unfixed_chunk_num", 3)
    kwargs.setdefault("unfixed_token_num", 0)
    return R2T2StreamSession(script, _tokenize, _detokenize, **kwargs)


def _pcm(samples: int, value: float = 0.1) -> np.ndarray:
    return np.full((samples,), value, dtype=np.float32)


def test_trace_records_sample_range_budget_and_commit() -> None:
    script = _Script(["hello", "hello world"])
    trace = R2T2StreamTrace()
    session = _session(script, trace=trace)

    session.feed(_pcm(_CHUNK_SAMPLES))
    session.feed(_pcm(_CHUNK_SAMPLES))

    assert [step.kind for step in trace.steps] == ["window", "window"]
    assert [step.index for step in trace.steps] == [0, 1]
    assert [(step.sample_start, step.sample_end) for step in trace.steps] == [
        (0, _CHUNK_SAMPLES),
        (_CHUNK_SAMPLES, 2 * _CHUNK_SAMPLES),
    ]
    assert [step.budget for step in trace.steps] == [2, 2]
    assert trace.steps[0].committed == "hello"
    assert trace.steps[0].delta == "hello"
    assert trace.steps[1].hypothesis == "hello world"
    assert trace.steps[1].committed == "hello world"
    assert trace.current is None


def test_finalize_records_one_tail_step_only_when_audio_remains() -> None:
    script = _Script(["hello"])
    trace = R2T2StreamTrace()
    session = _session(script, trace=trace)
    session.feed(_pcm(_CHUNK_SAMPLES))
    session.finalize()
    assert [step.kind for step in trace.steps] == ["window"]

    tail_trace = R2T2StreamTrace()
    tail_session = _session(_Script(["hello"]), trace=tail_trace)
    tail_session.feed(_pcm(_CHUNK_SAMPLES + 100))
    tail_session.finalize()
    assert [step.kind for step in tail_trace.steps] == ["window", "tail"]
    tail = tail_trace.steps[-1]
    assert (tail.sample_start, tail.sample_end) == (_CHUNK_SAMPLES, _CHUNK_SAMPLES + 100)


def test_trace_counts_every_pcm_copy() -> None:
    trace = R2T2StreamTrace()
    session = _session(_Script(["hello"]), trace=trace)

    session.feed(_pcm(_CHUNK_SAMPLES))
    session.feed(_pcm(_CHUNK_SAMPLES))

    # Buffer the incoming frame, then append the window once. History is
    # neither re-concatenated nor copied for the decoder.
    first_window = 2 * _CHUNK_SAMPLES
    second_window = 2 * _CHUNK_SAMPLES
    assert trace.copy_samples == first_window + second_window
    assert trace.copy_bytes == (first_window + second_window) * 4
    assert trace.steps[0].counter(PCM_SAMPLES_COPIED) == first_window
    assert trace.steps[0].counter(PCM_BYTES_COPIED) == first_window * 4
    assert trace.steps[1].counter(PCM_SAMPLES_COPIED) == second_window


def test_trace_is_absent_and_optional_by_default() -> None:
    script = _Script(["hello"])
    session = _session(script)
    assert session.trace is None
    update = session.feed(_pcm(_CHUNK_SAMPLES))
    assert session.trace is None
    assert update.hypothesis == "hello"

    traced = R2T2StreamTrace()
    traced_session = _session(_Script(["hello"]), trace=traced)
    traced_update = traced_session.feed(_pcm(_CHUNK_SAMPLES))
    assert traced_update == update
    assert len(traced.steps) == 1


def test_stage_accumulates_and_is_a_noop_without_metrics() -> None:
    with stage(None, "mel"):
        pass

    metrics = R2T2StepMetrics()
    with stage(metrics, "mel"):
        pass
    with stage(metrics, "mel"):
        pass

    assert set(metrics.stage_seconds) == {"mel"}
    assert metrics.stage_seconds["mel"] >= 0.0


def test_step_metrics_counters_accumulate_and_serialize() -> None:
    metrics = R2T2StepMetrics(index=3, kind="tail", sample_start=10, sample_end=12)
    metrics.add_stage("decode_step", 0.25)
    metrics.add_stage("decode_step", 0.5)
    metrics.add_counter(PCM_SAMPLES_COPIED, 100)
    metrics.add_counter(PCM_SAMPLES_COPIED, 50)

    payload = metrics.to_dict()
    assert payload["index"] == 3
    assert payload["kind"] == "tail"
    assert payload["stage_seconds"] == {"decode_step": 0.75}
    assert payload["counters"] == {PCM_SAMPLES_COPIED: 150}
    assert metrics.counter(PCM_SAMPLES_COPIED) == 150


def test_queue_simulation_keeps_up_with_arrival_rate() -> None:
    result = simulate_stream_queue([0.16, 0.32, 0.48], [0.10, 0.10, 0.10])

    assert result["windows"] == 3
    assert result["max_latency_seconds"] == pytest.approx(0.10)
    assert result["final_latency_seconds"] == pytest.approx(0.10)
    assert result["max_queue_wait_seconds"] == pytest.approx(0.0)
    assert result["max_backlog_windows"] == 0
    assert result["behind_windows"] == 0
    assert result["first_behind_seconds"] is None
    assert result["first_sustained_behind_seconds"] is None
    assert result["processing_audio_ratio"] == pytest.approx(0.30 / 0.48)
    assert result["step_p50_seconds"] == pytest.approx(0.10)
    assert result["step_p95_seconds"] == pytest.approx(0.10)


def test_queue_simulation_waits_for_arrival_before_starting() -> None:
    # Fast steps cannot bank time: each window starts only when its audio exists.
    result = simulate_stream_queue([0.16, 0.32], [0.01, 0.20])

    assert result["final_latency_seconds"] == pytest.approx(0.20)
    assert result["behind_windows"] == 0


def test_queue_simulation_reports_first_sustained_backlog() -> None:
    result = simulate_stream_queue(
        [0.16, 0.32, 0.48, 0.64],
        [0.30, 0.30, 0.10, 0.30],
    )

    assert result["behind_windows"] == 3
    assert result["first_behind_seconds"] == pytest.approx(0.32)
    assert result["first_sustained_behind_seconds"] == pytest.approx(0.32)
    assert result["max_queue_wait_seconds"] == pytest.approx(0.28)
    assert result["final_queue_wait_seconds"] == pytest.approx(0.22)
    assert result["final_latency_seconds"] == pytest.approx(1.16 - 0.64)
    assert result["max_backlog_windows"] == 2


def test_queue_simulation_detects_recovery() -> None:
    result = simulate_stream_queue(
        [0.16, 0.32, 0.48, 0.64],
        [0.40, 0.02, 0.02, 0.02],
    )

    assert result["behind_windows"] == 2
    assert result["first_behind_seconds"] == pytest.approx(0.32)
    assert result["first_sustained_behind_seconds"] is None
    assert result["final_queue_wait_seconds"] == pytest.approx(0.0)
    assert result["final_latency_seconds"] == pytest.approx(0.02)


def test_queue_simulation_validates_input() -> None:
    with pytest.raises(ValueError):
        simulate_stream_queue([0.10], [0.10, 0.20])
    with pytest.raises(ValueError):
        simulate_stream_queue([0.20, 0.10], [0.10, 0.10])


def test_queue_simulation_handles_empty_input() -> None:
    result = simulate_stream_queue([], [])

    assert result["windows"] == 0
    assert result["processing_audio_ratio"] is None
    assert result["first_behind_seconds"] is None
    assert result["step_p95_seconds"] is None
