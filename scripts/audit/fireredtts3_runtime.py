#!/usr/bin/env python3
"""Audit FireRedTTS3 sliding attention and end-to-end MLX runtime metrics."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    attention = commands.add_parser("attention")
    attention.add_argument("--lengths", type=int, nargs="+", required=True)
    attention.add_argument("--window", type=int, required=True)
    attention.add_argument("--isolated", action="store_true")
    attention.add_argument("--max-peak-growth", type=float, required=True)

    worker = commands.add_parser("attention-worker", help=argparse.SUPPRESS)
    worker.add_argument("--length", type=int, required=True)
    worker.add_argument("--window", type=int, required=True)

    profile = commands.add_parser("profile")
    profile.add_argument("--model-dir", type=Path, required=True)
    profile.add_argument("--reference-audio", type=Path, required=True)
    profile.add_argument("--reference-text", required=True)
    profile.add_argument("--text", required=True)
    profile.add_argument("--language", required=True)
    profile.add_argument("--seed", type=int, default=1234)
    profile.add_argument("--flow-steps", type=int, default=10)
    profile.add_argument("--guidance-scale", type=float, default=2.0)
    profile.add_argument("--stop-threshold", type=float, default=0.5)
    profile.add_argument("--max-audio-patches", type=int, default=40)
    profile.add_argument("--warmup-runs", type=int, default=1)
    profile.add_argument("--runs", type=int, required=True)
    profile.add_argument("--output", type=Path)
    profile.add_argument("--parity-baseline", type=Path)
    profile.add_argument("--baseline-core-seconds", type=float)
    profile.add_argument("--baseline-mlx-peak-gib", type=float)
    profile.add_argument("--min-candidate-improvement", type=float, default=0.05)
    profile.add_argument("--min-total-improvement", type=float, default=0.10)
    profile.add_argument("--max-other-regression", type=float, default=0.10)
    return parser


def _attention_worker(length: int, window: int) -> None:
    import mlx.core as mx

    from mlx_speech.models.qwen3_asr.text_decoder import (
        _sliding_window_attention,
    )

    mx.random.seed(29)
    query = mx.random.normal((1, 4, length, 64))
    key = mx.random.normal((1, 2, length, 64))
    value = mx.random.normal((1, 2, length, 64))
    mx.eval(query, key, value)
    mx.clear_cache()
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = _sliding_window_attention(
        query,
        key,
        value,
        scale=64**-0.5,
        sliding_window=window,
        query_block_size=window,
    )
    finite = bool(mx.all(mx.isfinite(output)).item())
    mx.synchronize()
    elapsed = time.perf_counter() - started
    peak_delta = max(0, int(mx.get_peak_memory()) - baseline)
    print(
        json.dumps(
            {
                "length": length,
                "window": window,
                "finite": finite,
                "seconds": elapsed,
                "baseline_active_bytes": baseline,
                "peak_delta_bytes": peak_delta,
            },
            sort_keys=True,
        )
    )


def _attention_parent(args: argparse.Namespace) -> None:
    if not args.isolated:
        raise ValueError("attention audit requires --isolated")
    if len(args.lengths) < 2 or sorted(args.lengths) != list(args.lengths):
        raise ValueError("attention lengths must contain two or more increasing values")
    results = []
    script = Path(__file__).resolve()
    for length in args.lengths:
        process = subprocess.run(
            [
                sys.executable,
                str(script),
                "attention-worker",
                "--length",
                str(length),
                "--window",
                str(args.window),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        results.append(json.loads(process.stdout))
    if not all(result["finite"] for result in results):
        raise RuntimeError("sliding attention produced non-finite output")
    first_peak = int(results[0]["peak_delta_bytes"])
    last_peak = int(results[-1]["peak_delta_bytes"])
    growth = last_peak / max(first_peak, 1)
    payload = {"results": results, "peak_growth": growth}
    print(json.dumps(payload, indent=2, sort_keys=True))
    if growth > args.max_peak_growth:
        raise RuntimeError(
            f"sliding attention peak growth {growth:.3f} exceeds "
            f"{args.max_peak_growth:.3f}"
        )


def _elapsed(started: float) -> tuple[float, float]:
    import mlx.core as mx

    mx.synchronize()
    now = time.perf_counter()
    return now - started, now


def _profile_once(model: Any, args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_speech.diagnostics import (
        process_peak_physical_footprint_bytes,
        snapshot_mlx_memory,
    )

    mx.reset_peak_memory()
    started = time.perf_counter()
    audio = model._load_reference(args.reference_audio, reference_sample_rate=None)
    multiple = model.redae.downsample_rate * model.core.config.patch_size
    prompt = model.redae.pad_audio(audio[None], multiple=multiple)
    mx.eval(prompt)
    reference_seconds, started = _elapsed(started)

    latents = model.redae.encode(prompt[0]).astype(mx.float32)
    mx.eval(latents)
    encode_seconds, started = _elapsed(started)

    speaker = model.speaker(prompt[0], sample_rate=model.sample_rate)
    mx.eval(speaker)
    speaker_seconds, started = _elapsed(started)

    token_ids = model.tokenizer.encode(
        language=args.language,
        reference_text=args.reference_text,
        text=args.text,
    )
    result = model.core.generate(
        speaker_embedding=speaker,
        text_tokens=mx.array([token_ids], dtype=mx.int32),
        prompt_latents=latents,
        flow_steps=args.flow_steps,
        guidance_scale=args.guidance_scale,
        stop_threshold=args.stop_threshold,
        max_generated_patches=args.max_audio_patches,
        seed=args.seed,
    )
    mx.eval(result.latents)
    core_seconds, started = _elapsed(started)

    decoded = model.redae.decode(result.latents).astype(mx.float32)
    waveform = decoded[0, int(prompt.shape[-1]) :]
    mx.eval(waveform)
    decode_seconds, _ = _elapsed(started)
    memory = snapshot_mlx_memory("profile")
    record = {
        "reference_seconds": reference_seconds,
        "redae_encode_seconds": encode_seconds,
        "speaker_seconds": speaker_seconds,
        "core_seconds": core_seconds,
        "redae_decode_seconds": decode_seconds,
        "generated_patches": result.generated_patches,
        "prompt_length": result.prompt_length,
        "cache_length": result.cache_length,
        "samples": int(waveform.size),
        "mlx_active_bytes": memory.active_bytes,
        "mlx_cache_bytes": memory.cache_bytes,
        "mlx_peak_bytes": memory.peak_bytes,
        "process_peak_bytes": process_peak_physical_footprint_bytes(),
    }
    del audio, prompt, latents, speaker, result, decoded, waveform
    gc.collect()
    mx.clear_cache()
    return record


def _median(records: list[dict[str, Any]], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def _range(records: list[dict[str, Any]], key: str) -> float:
    values = [float(record[key]) for record in records]
    return max(values) - min(values)


def _fraction_improvement(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / baseline


def _check_profile_gate(payload: dict[str, Any], args: argparse.Namespace) -> None:
    if args.parity_baseline is None:
        return
    if args.baseline_core_seconds is None or args.baseline_mlx_peak_gib is None:
        raise ValueError("historical core and MLX peak baselines are required")
    baseline = json.loads(args.parity_baseline.read_text(encoding="utf-8"))
    baseline_records = baseline["runs"]
    candidate_records = payload["runs"]
    baseline_core = float(baseline["medians"]["core_seconds"])
    candidate_core = float(payload["medians"]["core_seconds"])
    baseline_peak = float(baseline["medians"]["mlx_peak_bytes"])
    candidate_peak = float(payload["medians"]["mlx_peak_bytes"])
    improvements = {
        "core": _fraction_improvement(baseline_core, candidate_core),
        "peak": _fraction_improvement(baseline_peak, candidate_peak),
    }
    selected = max(improvements, key=improvements.get)
    selected_key = "core_seconds" if selected == "core" else "mlx_peak_bytes"
    median_delta = abs(
        float(baseline["medians"][selected_key])
        - float(payload["medians"][selected_key])
    )
    noise = max(
        _range(baseline_records, selected_key),
        _range(candidate_records, selected_key),
    )
    if improvements[selected] < args.min_candidate_improvement or median_delta <= noise:
        raise RuntimeError(
            "candidate improvement did not clear the parity baseline and run spread"
        )

    historical = {
        "core": float(args.baseline_core_seconds),
        "peak": float(args.baseline_mlx_peak_gib) * 1024**3,
    }
    current = {"core": candidate_core, "peak": candidate_peak}
    total = {
        name: _fraction_improvement(historical[name], current[name])
        for name in historical
    }
    total_selected = max(total, key=total.get)
    if total[total_selected] < args.min_total_improvement:
        raise RuntimeError("final runtime did not clear the historical improvement gate")
    other = "peak" if total_selected == "core" else "core"
    for comparison in (historical[other], {"core": baseline_core, "peak": baseline_peak}[other]):
        regression = (current[other] - comparison) / comparison
        if regression > args.max_other_regression:
            raise RuntimeError(f"{other} regressed by {regression:.3%}")
    payload["gate"] = {
        "candidate_improvements": improvements,
        "historical_improvements": total,
        "selected_candidate_metric": selected,
        "selected_historical_metric": total_selected,
        "selected_median_delta": median_delta,
        "selected_run_spread": noise,
    }


def _profile(args: argparse.Namespace) -> None:
    if args.runs < 3:
        raise ValueError("profile requires at least three measured runs")
    if args.warmup_runs < 1:
        raise ValueError("profile requires at least one excluded warmup")
    from mlx_speech import tts

    started = time.perf_counter()
    model = tts.load(str(args.model_dir))
    cold_load_seconds = time.perf_counter() - started
    for _ in range(args.warmup_runs):
        _profile_once(model, args)
    records = [_profile_once(model, args) for _ in range(args.runs)]
    median_keys = (
        "reference_seconds",
        "redae_encode_seconds",
        "speaker_seconds",
        "core_seconds",
        "redae_decode_seconds",
        "mlx_active_bytes",
        "mlx_cache_bytes",
        "mlx_peak_bytes",
        "process_peak_bytes",
    )
    payload: dict[str, Any] = {
        "cold_load_seconds": cold_load_seconds,
        "warmup_runs": args.warmup_runs,
        "runs": records,
        "medians": {key: _median(records, key) for key in median_keys},
    }
    _check_profile_gate(payload, args)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")


def main() -> None:
    args = _parser().parse_args()
    if args.command == "attention-worker":
        _attention_worker(args.length, args.window)
    elif args.command == "attention":
        _attention_parent(args)
    else:
        _profile(args)


if __name__ == "__main__":
    main()
