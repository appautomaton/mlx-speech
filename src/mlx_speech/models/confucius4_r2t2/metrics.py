"""Private per-step instrumentation for R2T2 streaming.

The tracing types here are opt-in. ``R2T2StreamSession`` and its ASR adapter
do no extra work when no trace is attached, which is the production path; the
benchmark harness attaches one only in profiling runs.

The record separates *executed* work from returned tensor shapes, because the
streaming performance plan gates on work that was actually computed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

# Counter names recorded on ``R2T2StepMetrics.counters``.
MEL_FRAMES = "mel_frames"
ENCODER_BLOCKS = "encoder_blocks"
ENCODER_TOKENS = "encoder_tokens"
PREFILL_POSITIONS = "prefill_positions"
VOCAB_POSITIONS = "vocab_positions"
DECODE_STEPS = "decode_steps"
CACHE_ALLOCATIONS = "cache_allocations"
PCM_SAMPLES_COPIED = "pcm_samples_copied"
PCM_BYTES_COPIED = "pcm_bytes_copied"


@dataclass
class R2T2StepMetrics:
    """One profiled window or final tail."""

    index: int = -1
    kind: str = "window"
    sample_start: int = 0
    sample_end: int = 0
    budget: int = 0
    stage_seconds: dict[str, float] = field(default_factory=dict)
    counters: dict[str, int] = field(default_factory=dict)
    prompt_tokens: int = 0
    audio_tokens: int = 0
    mel_frames: int = 0
    completion: str = ""
    generated_ids: tuple[int, ...] = ()
    top_logits: tuple[tuple[int, float], ...] = ()
    top1_minus_top2: float | None = None
    hypothesis: str = ""
    committed: str = ""
    delta: str = ""
    llm_seconds: float = 0.0
    step_seconds: float = 0.0

    def add_stage(self, name: str, seconds: float) -> None:
        self.stage_seconds[name] = self.stage_seconds.get(name, 0.0) + float(seconds)

    def add_counter(self, name: str, value: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + int(value)

    def counter(self, name: str) -> int:
        return int(self.counters.get(name, 0))

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "index": self.index,
            "kind": self.kind,
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "budget": self.budget,
            "prompt_tokens": self.prompt_tokens,
            "audio_tokens": self.audio_tokens,
            "mel_frames": self.mel_frames,
            "generated_ids": list(self.generated_ids),
            "completion": self.completion,
            "top_logits": [[int(token), float(logit)] for token, logit in self.top_logits],
            "top1_minus_top2": self.top1_minus_top2,
            "hypothesis": self.hypothesis,
            "committed": self.committed,
            "delta": self.delta,
            "llm_seconds": self.llm_seconds,
            "step_seconds": self.step_seconds,
            "stage_seconds": {k: float(v) for k, v in self.stage_seconds.items()},
            "counters": {k: int(v) for k, v in self.counters.items()},
        }
        return payload


class R2T2StreamTrace:
    """Per-session collector. ``current`` is set by the session for one step."""

    def __init__(self) -> None:
        self.steps: list[R2T2StepMetrics] = []
        self.current: R2T2StepMetrics | None = None
        self.copy_samples = 0
        self.copy_bytes = 0
        # Copies made before any step was open (for example a ``feed`` buffer
        # append) belong to the first step they enable.
        self.pending_copy_samples = 0
        self.pending_copy_bytes = 0

    def begin(
        self,
        *,
        index: int,
        kind: str,
        sample_start: int,
        sample_end: int,
    ) -> R2T2StepMetrics:
        metrics = R2T2StepMetrics(
            index=index,
            kind=kind,
            sample_start=sample_start,
            sample_end=sample_end,
        )
        if self.pending_copy_samples:
            metrics.add_counter(PCM_SAMPLES_COPIED, self.pending_copy_samples)
            metrics.add_counter(PCM_BYTES_COPIED, self.pending_copy_bytes)
            self.pending_copy_samples = 0
            self.pending_copy_bytes = 0
        self.steps.append(metrics)
        self.current = metrics
        return metrics

    def end(self) -> None:
        self.current = None

    def count_copy(self, samples: int, itemsize: int, *, defer: bool = False) -> None:
        """Record one buffer/array copy of ``samples`` PCM samples.

        ``defer`` holds the copy aside until the next ``begin`` so an incoming
        ``feed`` chunk lands on the first window it enables instead of nowhere.
        """

        samples = int(samples)
        if samples <= 0:
            return
        copied_bytes = samples * int(itemsize)
        self.copy_samples += samples
        self.copy_bytes += copied_bytes
        if defer:
            self.pending_copy_samples += samples
            self.pending_copy_bytes += copied_bytes
            return
        if self.current is not None:
            self.current.add_counter(PCM_SAMPLES_COPIED, samples)
            self.current.add_counter(PCM_BYTES_COPIED, copied_bytes)


@contextmanager
def stage(metrics: R2T2StepMetrics | None, name: str) -> Iterator[None]:
    """Time one stage; a no-op context manager when tracing is off."""

    if metrics is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        metrics.add_stage(name, time.perf_counter() - started)


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile needs at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile; exported for the harness and tests."""

    return _percentile(values, fraction)


def simulate_stream_queue(
    arrival_seconds: Sequence[float],
    step_seconds: Sequence[float],
) -> dict[str, Any]:
    """Single-server FIFO queue at the real-time arrival rate.

    ``arrival_seconds[i]`` is when window ``i`` becomes available (stream
    time, non-decreasing); ``step_seconds[i]`` is its measured processing
    time. A window starts at ``max(arrival, previous finish)``. ``latency`` is
    finish minus arrival; ``queue wait`` is start minus arrival. A window is
    *behind* when it waits for the previous one.
    """

    if len(arrival_seconds) != len(step_seconds):
        raise ValueError("arrival_seconds and step_seconds must have equal length")
    if not arrival_seconds:
        return {
            "windows": 0,
            "processing_seconds": 0.0,
            "audio_seconds": 0.0,
            "processing_audio_ratio": None,
            "max_latency_seconds": 0.0,
            "final_latency_seconds": 0.0,
            "max_queue_wait_seconds": 0.0,
            "final_queue_wait_seconds": 0.0,
            "max_backlog_windows": 0,
            "behind_windows": 0,
            "first_behind_seconds": None,
            "first_sustained_behind_seconds": None,
            "step_p50_seconds": None,
            "step_p95_seconds": None,
        }

    arrivals = [float(value) for value in arrival_seconds]
    steps = [float(value) for value in step_seconds]
    for previous, current in zip(arrivals, arrivals[1:]):
        if current < previous:
            raise ValueError("arrival_seconds must be non-decreasing")

    starts: list[float] = []
    finishes: list[float] = []
    finish = float("-inf")
    for arrival, step in zip(arrivals, steps, strict=True):
        start = max(arrival, finish)
        finish = start + step
        starts.append(start)
        finishes.append(finish)

    waits = [start - arrival for start, arrival in zip(starts, arrivals, strict=True)]
    latencies = [done - arrival for done, arrival in zip(finishes, arrivals, strict=True)]

    # Windows that have arrived but not started, sampled at each arrival.
    backlog = 0
    first_unstarted = 0
    for index, arrival in enumerate(arrivals):
        while first_unstarted <= index and starts[first_unstarted] <= arrival:
            first_unstarted += 1
        backlog = max(backlog, index + 1 - first_unstarted)

    behind = [index for index, wait in enumerate(waits) if wait > 0.0]
    sustained: float | None = None
    for index in behind:
        if all(wait > 0.0 for wait in waits[index:]):
            sustained = arrivals[index]
            break

    audio_seconds = arrivals[-1]
    processing = float(sum(steps))
    return {
        "windows": len(steps),
        "processing_seconds": processing,
        "audio_seconds": audio_seconds,
        "processing_audio_ratio": (processing / audio_seconds) if audio_seconds else None,
        "max_latency_seconds": max(latencies),
        "final_latency_seconds": latencies[-1],
        "max_queue_wait_seconds": max(waits),
        "final_queue_wait_seconds": waits[-1],
        "max_backlog_windows": backlog,
        "behind_windows": len(behind),
        "first_behind_seconds": (arrivals[behind[0]] if behind else None),
        "first_sustained_behind_seconds": sustained,
        "step_p50_seconds": _percentile(steps, 0.50),
        "step_p95_seconds": _percentile(steps, 0.95),
    }


__all__ = [
    "CACHE_ALLOCATIONS",
    "DECODE_STEPS",
    "ENCODER_BLOCKS",
    "ENCODER_TOKENS",
    "MEL_FRAMES",
    "PCM_BYTES_COPIED",
    "PCM_SAMPLES_COPIED",
    "PREFILL_POSITIONS",
    "R2T2StepMetrics",
    "R2T2StreamTrace",
    "VOCAB_POSITIONS",
    "percentile",
    "simulate_stream_queue",
    "stage",
]
