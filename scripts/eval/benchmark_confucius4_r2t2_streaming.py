#!/usr/bin/env python3
"""Baseline harness for Confucius4-R2T2 streaming inference cost.

Drives one R2T2 streaming session window by window over real fixtures and
records, per window: wall latency, stage times, executed-work counters, and the
decoder capture (prompt tokens, budget, generated ids, top-k logits, top-1/top-2
margin, hypothesis, committed text).

Two measurements per length:

* ``production`` — trace off, no ``mx.eval`` barriers. Per-window wall latency,
  a queue simulation per pass, and one whole-file ``feed`` run for comparison.
* ``profiled`` — trace on, ``mx.eval`` barriers between stages, so the harness
  can attribute cost to mel / audio tower / prompt / embedding / prefill body /
  vocabulary projection / decode.

A single-server queue simulation turns arrival times and step latencies into
backlog over time and the processing/audio ratio.

This is timing evidence, not parity acceptance. Entry 1 of the evidence log in
``.agents/plans/confucius4-r2t2-live-cost.md`` quotes its numbers.

    .venv/bin/python scripts/eval/benchmark_confucius4_r2t2_streaming.py
    .venv/bin/python scripts/eval/benchmark_confucius4_r2t2_streaming.py \\
        --chunk-ms 160 80 --lengths 15 60 --include-informational
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_RATE = 16_000
DEFAULT_MODEL = "models/netease/confucius4_r2t2/mlx-bf16"
DEFAULT_FIXTURES = (
    "models/netease/confucius4_r2t2/samples/test.wav",
    "outputs/source/hank_hill_ref.wav",
    "outputs/source/peggy_hill_ref.wav",
    "outputs/source/donald_trump_ref.wav",
)
DEFAULT_LENGTHS = (1, 5, 8, 15, 30, 60)
INFORMATIONAL_LENGTHS = (120,)
DEFAULT_CHUNK_MS = (160,)
DEFAULT_LOOKAHEAD_MS = 160
DEFAULT_REPEATS = 3
DEFAULT_PROFILE_REPEATS = 1
STAGE_ORDER = (
    "mel",
    "audio_tower",
    "prompt_build",
    "embed",
    "prefill_body",
    "vocab",
    "decode_step",
)
COUNTER_ORDER = (
    "mel_frames",
    "encoder_blocks",
    "encoder_tokens",
    "prefill_positions",
    "vocab_positions",
    "decode_steps",
    "cache_allocations",
)


# --------------------------------------------------------------------------- #
# Pure helpers (no MLX import at module scope so they stay unit-testable)
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    """Linear-interpolation percentile over a non-empty list."""

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


def build_concatenated_audio(clips: list[np.ndarray], *, target_samples: int) -> np.ndarray:
    """Concatenate distinct clips round-robin and truncate to the target."""

    if not clips:
        raise ValueError("build_concatenated_audio needs at least one clip")
    if target_samples <= 0:
        raise ValueError("target_samples must be positive")
    parts: list[np.ndarray] = []
    total = 0
    index = 0
    while total < target_samples:
        clip = clips[index % len(clips)]
        if clip.size == 0:
            raise ValueError("clips must be non-empty")
        parts.append(clip)
        total += int(clip.size)
        index += 1
    return np.concatenate(parts)[:target_samples].astype(np.float32, copy=False)


def ms_to_samples(milliseconds: int) -> int:
    return int(round(float(milliseconds) / 1000.0 * SAMPLE_RATE))


def window_plan(
    total_samples: int,
    *,
    chunk_samples: int,
    lookahead_samples: int,
) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
    """Sample ranges for the published schedule: the first window adds lookahead."""

    plan: list[tuple[int, int]] = []
    offset = 0
    awaiting_first = True
    while True:
        window = chunk_samples + lookahead_samples if awaiting_first else chunk_samples
        if total_samples - offset >= window:
            plan.append((offset, offset + window))
            offset += window
            awaiting_first = False
        else:
            break
    tail = (offset, total_samples) if total_samples > offset else None
    return plan, tail


def arrival_seconds(
    total_samples: int,
    *,
    chunk_samples: int,
    lookahead_samples: int,
) -> list[float]:
    """When each scheduled window's audio became available, in stream seconds."""

    plan, tail = window_plan(
        total_samples,
        chunk_samples=chunk_samples,
        lookahead_samples=lookahead_samples,
    )
    arrivals = [end / SAMPLE_RATE for _, end in plan]
    if tail is not None:
        arrivals.append(tail[1] / SAMPLE_RATE)
    return arrivals


def summarize(values: list[float]) -> dict[str, Any]:
    """Median / p95 / max / spread of a latency list in seconds."""

    if not values:
        return {
            "count": 0,
            "p50_seconds": None,
            "p95_seconds": None,
            "max_seconds": None,
            "mean_seconds": None,
            "spread_seconds": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "p50_seconds": percentile(ordered, 0.50),
        "p95_seconds": percentile(ordered, 0.95),
        "max_seconds": ordered[-1],
        "mean_seconds": float(sum(ordered) / len(ordered)),
        "spread_seconds": ordered[-1] - ordered[0],
    }


def median_pass(queue_runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the pass with the median processing/audio ratio."""

    if not queue_runs:
        raise ValueError("median_pass needs at least one queue run")
    ratios = [float(run["processing_audio_ratio"] or 0.0) for run in queue_runs]
    order = sorted(range(len(ratios)), key=lambda index: ratios[index])
    return queue_runs[order[len(order) // 2]]


# --------------------------------------------------------------------------- #
# Session driving
# --------------------------------------------------------------------------- #
def drive_windowed(
    session,
    audio: np.ndarray,
    *,
    chunk_samples: int,
    lookahead_samples: int,
) -> dict[str, Any]:
    """Feed exactly one scheduled window per ``feed`` call, then finalize."""

    plan, tail = window_plan(
        int(audio.shape[0]),
        chunk_samples=chunk_samples,
        lookahead_samples=lookahead_samples,
    )
    records: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(plan):
        started = time.perf_counter()
        session.feed(audio[start:end])
        records.append(
            {
                "index": index,
                "kind": "window",
                "sample_start": start,
                "sample_end": end,
                "arrival_seconds": end / SAMPLE_RATE,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    if tail is not None:
        session.feed(audio[tail[0]:tail[1]])
    started = time.perf_counter()
    final = session.finalize()
    finalize_seconds = time.perf_counter() - started
    if tail is not None:
        records.append(
            {
                "index": len(records),
                "kind": "tail",
                "sample_start": tail[0],
                "sample_end": tail[1],
                "arrival_seconds": tail[1] / SAMPLE_RATE,
                "wall_seconds": finalize_seconds,
            }
        )
    return {
        "records": records,
        "finalize_seconds": finalize_seconds,
        "text": final.text,
        "language": final.language,
        "tail": tail,
    }


def drive_whole_file(session, audio: np.ndarray) -> float:
    """One ``feed`` with the whole file, then ``finalize``."""

    started = time.perf_counter()
    session.feed(audio)
    session.finalize()
    return time.perf_counter() - started


def assert_schedule(records: list[dict[str, Any]], capture: list[dict[str, Any]]) -> None:
    """The harness plan must match the sample ranges the session reported."""

    if len(records) != len(capture):
        raise AssertionError(
            f"window plan has {len(records)} steps, session traced {len(capture)}"
        )
    for record, step in zip(records, capture, strict=True):
        planned = (record["sample_start"], record["sample_end"])
        traced = (step["sample_start"], step["sample_end"])
        if planned != traced:
            raise AssertionError(f"window schedule mismatch: plan {planned} vs session {traced}")


# --------------------------------------------------------------------------- #
# One length × one schedule
# --------------------------------------------------------------------------- #
def run_case(
    adapter,
    audio: np.ndarray,
    *,
    chunk_ms: int,
    lookahead_ms: int,
    language: str | None,
    repeats: int,
    profile_repeats: int,
) -> dict[str, Any]:
    from mlx_speech.models.confucius4_r2t2.metrics import simulate_stream_queue

    chunk_samples = ms_to_samples(chunk_ms)
    lookahead_samples = ms_to_samples(lookahead_ms)
    sessions = max(1, repeats)

    arrivals = arrival_seconds(
        int(audio.shape[0]),
        chunk_samples=chunk_samples,
        lookahead_samples=lookahead_samples,
    )
    window_seconds: list[float] = []
    tail_seconds: list[float] = []
    queue_runs: list[dict[str, Any]] = []
    pass_steps: list[list[float]] = []
    whole_seconds: list[float] = []
    text = ""
    for _ in range(sessions):
        session = adapter.stream_session(
            chunk_ms=chunk_ms,
            lookahead_ms=lookahead_ms,
            language=language,
        )
        run = drive_windowed(
            session,
            audio,
            chunk_samples=chunk_samples,
            lookahead_samples=lookahead_samples,
        )
        steps = [record["wall_seconds"] for record in run["records"]]
        window_seconds.extend(
            record["wall_seconds"] for record in run["records"] if record["kind"] == "window"
        )
        tail_seconds.extend(
            record["wall_seconds"] for record in run["records"] if record["kind"] == "tail"
        )
        pass_steps.append(steps)
        queue_runs.append(simulate_stream_queue(arrivals, steps))
        text = run["text"]

        whole_session = adapter.stream_session(
            chunk_ms=chunk_ms,
            lookahead_ms=lookahead_ms,
            language=language,
        )
        whole_seconds.append(drive_whole_file(whole_session, audio))

    queue = median_pass(queue_runs)
    profile = None
    if profile_repeats > 0:
        profile = _profile_case(
            adapter,
            audio,
            chunk_ms=chunk_ms,
            lookahead_ms=lookahead_ms,
            language=language,
            repeats=int(profile_repeats),
        )
        assert_schedule(profile["records"], profile["capture"])
        profile["final_text_matches_production"] = bool(
            profile["capture"] and profile["capture"][-1]["committed"] == text
        )

    all_steps = window_seconds + tail_seconds
    return {
        "chunk_ms": chunk_ms,
        "lookahead_ms": lookahead_ms,
        "language": language,
        "audio_samples": int(audio.shape[0]),
        "audio_seconds": float(audio.shape[0]) / SAMPLE_RATE,
        "windows": len(arrivals),
        "repeats": sessions,
        "production": {
            "window_step": summarize(window_seconds),
            "tail_step": summarize(tail_seconds),
            "all_steps": summarize(all_steps),
            "processing_seconds": float(sum(all_steps)) / sessions,
            "whole_file_seconds": whole_seconds,
            "whole_file_p50_seconds": summarize(whole_seconds)["p50_seconds"],
            "queue_runs": [
                {
                    "processing_audio_ratio": run["processing_audio_ratio"],
                    "max_latency_seconds": run["max_latency_seconds"],
                    "final_latency_seconds": run["final_latency_seconds"],
                    "max_queue_wait_seconds": run["max_queue_wait_seconds"],
                    "max_backlog_windows": run["max_backlog_windows"],
                }
                for run in queue_runs
            ],
            "step_seconds_per_pass": pass_steps,
        },
        "queue": queue,
        "profile": profile,
        "final_text": text,
    }


def _profile_case(
    adapter,
    audio: np.ndarray,
    *,
    chunk_ms: int,
    lookahead_ms: int,
    language: str | None,
    repeats: int,
) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_speech.models.confucius4_r2t2.metrics import R2T2StreamTrace

    chunk_samples = ms_to_samples(chunk_ms)
    lookahead_samples = ms_to_samples(lookahead_ms)
    stage_totals: dict[str, float] = {}
    counter_totals: dict[str, int] = {}
    step_totals_seconds = 0.0
    llm_totals_seconds = 0.0
    peak_bytes = 0
    records: list[dict[str, Any]] = []
    capture: list[dict[str, Any]] = []
    for _ in range(max(1, repeats)):
        trace = R2T2StreamTrace()
        session = adapter.stream_session(
            chunk_ms=chunk_ms,
            lookahead_ms=lookahead_ms,
            language=language,
            trace=trace,
        )
        mx.reset_peak_memory()
        mx.clear_cache()
        baseline_bytes = int(mx.get_active_memory())
        run = drive_windowed(
            session,
            audio,
            chunk_samples=chunk_samples,
            lookahead_samples=lookahead_samples,
        )
        peak_bytes = max(peak_bytes, int(mx.get_peak_memory()) - baseline_bytes)
        records = run["records"]
        capture = [step.to_dict() for step in trace.steps]
        for step in capture:
            for name, seconds in step["stage_seconds"].items():
                stage_totals[name] = stage_totals.get(name, 0.0) + float(seconds)
            for name, value in step["counters"].items():
                counter_totals[name] = counter_totals.get(name, 0) + int(value)
            step_totals_seconds += float(step["step_seconds"])
            llm_totals_seconds += float(step["llm_seconds"])
        counter_totals["pcm_samples_copied"] = int(trace.copy_samples)
        counter_totals["pcm_bytes_copied"] = int(trace.copy_bytes)

    return {
        "records": records,
        "capture": capture,
        "stage_totals_seconds": stage_totals,
        "counter_totals": counter_totals,
        "step_seconds": step_totals_seconds,
        "llm_seconds": llm_totals_seconds,
        "text_work_seconds": step_totals_seconds - llm_totals_seconds,
        "profiled_mlx_peak_bytes": peak_bytes,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def classify(case: dict[str, Any], *, step_target_seconds: float) -> dict[str, Any]:
    """Target: p95 step within the chunk interval, and no window ever waits
    longer than one chunk interval behind the previous one."""

    queue = case["queue"]
    p95 = queue["step_p95_seconds"]
    max_wait = float(queue["max_queue_wait_seconds"])
    latency_ok = p95 is not None and float(p95) <= step_target_seconds
    backlog_ok = max_wait <= step_target_seconds
    return {
        "step_target_seconds": step_target_seconds,
        "p95_over_target": not latency_ok,
        "max_queue_wait_seconds": max_wait,
        "backlog_over_target": not backlog_ok,
        "target_met": bool(latency_ok and backlog_ok),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    import mlx_speech
    from mlx_speech.audio import load_audio

    model_dir = Path(args.model_dir)
    fixture_paths = [Path(path) for path in args.fixtures]
    missing = [str(path) for path in fixture_paths if not path.exists()]
    if missing:
        raise SystemExit(f"missing fixtures: {', '.join(missing)}")

    clips: list[np.ndarray] = []
    fixtures: list[dict[str, str]] = []
    for path in fixture_paths:
        samples, _ = load_audio(path, sample_rate=SAMPLE_RATE, mono=True)
        flattened = np.asarray(samples, dtype=np.float32).reshape(-1)
        clips.append(flattened)
        fixtures.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "seconds": f"{flattened.shape[0] / SAMPLE_RATE:.3f}",
            }
        )

    adapter = mlx_speech.asr.load(str(model_dir))

    # One short streaming pass so the first measured length is not the very
    # first Metal dispatch of the process.
    warm = adapter.stream_session(
        chunk_ms=int(args.chunk_ms[0]),
        lookahead_ms=int(args.lookahead_ms),
        language=args.language,
    )
    warm.feed(np.zeros(SAMPLE_RATE, dtype=np.float32))
    warm.finalize()

    lengths = sorted({int(value) for value in args.lengths})
    if args.include_informational:
        lengths = sorted(set(lengths) | set(INFORMATIONAL_LENGTHS))

    cases: list[dict[str, Any]] = []
    for chunk_ms in args.chunk_ms:
        for seconds in lengths:
            audio = build_concatenated_audio(clips, target_samples=int(round(seconds * SAMPLE_RATE)))
            started = time.perf_counter()
            case = run_case(
                adapter,
                audio,
                chunk_ms=int(chunk_ms),
                lookahead_ms=int(args.lookahead_ms),
                language=args.language,
                repeats=int(args.repeats),
                profile_repeats=int(args.profile_repeats),
            )
            case["seconds"] = seconds
            case["informational"] = seconds in INFORMATIONAL_LENGTHS
            case["wall_seconds"] = time.perf_counter() - started
            case["classification"] = classify(
                case, step_target_seconds=int(chunk_ms) / 1000.0
            )
            cases.append(case)
            print(
                f"  {chunk_ms:>5} ms / {seconds:>4} s: "
                f"p50={_ms(case['production']['all_steps']['p50_seconds'])} "
                f"p95={_ms(case['production']['all_steps']['p95_seconds'])} "
                f"ratio={_ratio(case['queue']['processing_audio_ratio'])} "
                f"max_wait={_ms(case['queue']['max_queue_wait_seconds'])} "
                f"target={'met' if case['classification']['target_met'] else 'MISSED'}"
                f"  [{case['wall_seconds']:.1f}s]",
                flush=True,
            )

    crossover = None
    for case in cases:
        if case["informational"]:
            continue
        if not case["classification"]["target_met"]:
            crossover = case["seconds"]
            break

    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(model_dir),
        "fixtures": fixtures,
        "settings": {
            "chunk_ms": [int(value) for value in args.chunk_ms],
            "lookahead_ms": int(args.lookahead_ms),
            "language": args.language,
            "repeats": int(args.repeats),
            "profile_repeats": int(args.profile_repeats),
            "lengths": lengths,
            "step_target": "chunk interval (p95 step and max queue wait)",
        },
        "environment": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "mlx": getattr(mx, "__version__", "unknown"),
            "cpu_count": os.cpu_count(),
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "device": str(mx.default_device()),
        },
        "crossover_seconds": crossover,
        "cases": cases,
    }


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000.0:6.1f}ms"


def _ratio(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_tables(report: dict[str, Any]) -> None:
    print("\n== production step latency (trace off, no eval barriers) ==")
    header = (
        f"{'chunk':>7}{'audio':>7}{'windows':>9}{'p50':>10}{'p95':>10}{'max':>10}"
        f"{'proc s':>9}{'ratio':>8}{'max wait':>10}{'backlog':>9}"
        f"{'stop->final':>13}{'target':>9}"
    )
    print(header)
    print("-" * len(header))
    for case in report["cases"]:
        stats = case["production"]["all_steps"]
        print(
            f"{case['chunk_ms']:>7}{case['seconds']:>7}{case['windows']:>9}"
            f"{_ms(stats['p50_seconds']):>10}{_ms(stats['p95_seconds']):>10}"
            f"{_ms(stats['max_seconds']):>10}"
            f"{case['production']['processing_seconds']:>9.1f}"
            f"{_ratio(case['queue']['processing_audio_ratio']):>8}"
            f"{_ms(case['queue']['max_queue_wait_seconds']):>10}"
            f"{case['queue']['max_backlog_windows']:>9}"
            f"{_ms(case['queue']['final_latency_seconds']):>13}"
            f"{('met' if case['classification']['target_met'] else 'MISSED'):>9}"
        )

    print("\n== profiled stage totals (eval barriers between stages) ==")
    columns = ("chunk", "audio", *STAGE_ORDER, "text")
    header = "".join(f"{name:>10}" for name in columns)
    print(header)
    print("-" * len(header))
    for case in report["cases"]:
        profile = case["profile"]
        if not profile:
            continue
        stages = profile["stage_totals_seconds"]
        cells = "".join(f"{_ms(stages.get(name)):>10}" for name in STAGE_ORDER)
        print(
            f"{case['chunk_ms']:>10}{case['seconds']:>10}"
            f"{cells}{_ms(profile['text_work_seconds']):>10}"
        )

    print("\n== executed work totals (profile run) ==")
    columns = ("chunk", "audio", *COUNTER_ORDER)
    header = "".join(f"{name:>15}" for name in columns)
    print(header)
    print("-" * len(header))
    for case in report["cases"]:
        profile = case["profile"]
        if not profile:
            continue
        counters = profile["counter_totals"]
        cells = "".join(f"{counters.get(name, 0):>15}" for name in COUNTER_ORDER)
        print(f"{case['chunk_ms']:>15}{case['seconds']:>15}{cells}")

    crossover = report["crossover_seconds"]
    print(
        "\ncrossover length (first tested length missing the target): "
        + (f"{crossover} s" if crossover is not None else "none within the tested lengths")
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL)
    parser.add_argument("--fixtures", nargs="+", default=list(DEFAULT_FIXTURES))
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument("--include-informational", action="store_true")
    parser.add_argument("--chunk-ms", type=int, nargs="+", default=list(DEFAULT_CHUNK_MS))
    parser.add_argument("--lookahead-ms", type=int, default=DEFAULT_LOOKAHEAD_MS)
    parser.add_argument("--language", default=None)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--profile-repeats", type=int, default=DEFAULT_PROFILE_REPEATS)
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run_benchmark(args)
    print_tables(report)
    report_path = Path(
        args.report
        or (Path(os.environ.get("TMPDIR", "/tmp")) / "r2t2_streaming_baseline.json")
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nreport: {report_path}")


if __name__ == "__main__":
    main()
