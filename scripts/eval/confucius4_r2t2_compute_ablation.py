#!/usr/bin/env python3
"""Phase 2 compute ablation for Confucius4-R2T2 streaming.

Runs the same windowed streaming pass under several step-compute presets and
reports production step latency for each. Presets are interleaved inside every
repeat (rotating order) so GPU contention and drift hit all presets alike;
compare presets within one report, not across reports.

This is timing evidence only. Precision is judged separately by
``confucius4_r2t2_precision_gate.py``.

    .venv/bin/python scripts/eval/confucius4_r2t2_compute_ablation.py
    .venv/bin/python scripts/eval/confucius4_r2t2_compute_ablation.py \\
        --lengths 15 60 --presets reference native_linear all --repeats 5
"""

from __future__ import annotations

import argparse
import importlib.util
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
DEFAULT_LENGTHS = (1, 5, 15, 30, 60)


def _load_benchmark_module():
    path = Path(__file__).resolve().parent / "benchmark_confucius4_r2t2_streaming.py"
    spec = importlib.util.spec_from_file_location("_r2t2_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def preset_table() -> dict[str, Any]:
    """Named step-compute presets: each phase 2 item alone, then combinations."""

    from mlx_speech.asr._adapters.confucius4_r2t2 import R2T2StreamCompute
    from mlx_speech.models.qwen3_asr.text_decoder import Qwen3ASRDecodeCompute

    def decoder(**flags) -> Qwen3ASRDecodeCompute:
        return Qwen3ASRDecodeCompute(**flags)

    full_decoder = decoder(native_linear=True, native_lm_head=True, fused_attention=True)
    return {
        "reference": R2T2StreamCompute(),
        "native_linear": R2T2StreamCompute(decoder=decoder(native_linear=True)),
        "native_lm_head": R2T2StreamCompute(decoder=decoder(native_lm_head=True)),
        "last_logits": R2T2StreamCompute(last_logits_only=True),
        "fused_attention": R2T2StreamCompute(decoder=decoder(fused_attention=True)),
        "audio_bf16": R2T2StreamCompute(audio_bf16=True),
        "residual_bf16": R2T2StreamCompute(residual_bf16=True),
        "decoder_native": R2T2StreamCompute(decoder=full_decoder, last_logits_only=True),
        "decoder_native_fp32_head": R2T2StreamCompute(
            decoder=decoder(native_linear=True, fused_attention=True),
            last_logits_only=True,
        ),
        "all": R2T2StreamCompute(
            decoder=full_decoder,
            last_logits_only=True,
            audio_bf16=True,
            residual_bf16=True,
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    import mlx_speech
    from mlx_speech.audio import load_audio
    from mlx_speech.models.confucius4_r2t2.metrics import simulate_stream_queue

    bench = _load_benchmark_module()
    presets = preset_table()
    unknown = [name for name in args.presets if name not in presets]
    if unknown:
        raise SystemExit(f"unknown presets: {', '.join(unknown)}; known: {', '.join(presets)}")

    clips = []
    fixtures = []
    for path in (Path(item) for item in args.fixtures):
        samples, _ = load_audio(path, sample_rate=SAMPLE_RATE, mono=True)
        clip = np.asarray(samples, dtype=np.float32).reshape(-1)
        clips.append(clip)
        fixtures.append({"path": str(path), "sha256": bench.sha256_file(path)})

    adapter = mlx_speech.asr.load(str(args.model_dir))
    chunk = bench.ms_to_samples(args.chunk_ms)
    lookahead = bench.ms_to_samples(args.lookahead_ms)

    def one_pass(name: str, audio: np.ndarray) -> dict[str, Any]:
        session = adapter.stream_session(
            chunk_ms=args.chunk_ms,
            lookahead_ms=args.lookahead_ms,
            language=args.language,
            compute=presets[name],
        )
        return bench.drive_windowed(
            session, audio, chunk_samples=chunk, lookahead_samples=lookahead
        )

    warm = bench.build_concatenated_audio(clips, target_samples=SAMPLE_RATE)
    for name in args.presets:
        one_pass(name, warm)

    cases = []
    for seconds in sorted(set(args.lengths)):
        audio = bench.build_concatenated_audio(
            clips, target_samples=int(round(seconds * SAMPLE_RATE))
        )
        arrivals = bench.arrival_seconds(
            int(audio.shape[0]), chunk_samples=chunk, lookahead_samples=lookahead
        )
        steps: dict[str, list[float]] = {name: [] for name in args.presets}
        passes: dict[str, list[dict[str, Any]]] = {name: [] for name in args.presets}
        texts: dict[str, str] = {}
        order = list(args.presets)
        for repeat in range(args.repeats):
            rotated = order[repeat % len(order):] + order[: repeat % len(order)]
            for name in rotated:
                result = one_pass(name, audio)
                walls = [record["wall_seconds"] for record in result["records"]]
                steps[name].extend(walls)
                passes[name].append(simulate_stream_queue(arrivals, walls))
                texts[name] = result["text"]
        per_preset = {}
        for name in args.presets:
            summary = bench.summarize(steps[name])
            queue = bench.median_pass(passes[name])
            per_preset[name] = {
                "step": summary,
                "queue": queue,
                "final_text": texts[name],
                "text_matches_reference": (
                    texts[name] == texts.get("reference") if "reference" in texts else None
                ),
            }
        cases.append({"seconds": seconds, "windows": len(arrivals), "presets": per_preset})
        _print_case(cases[-1], args.presets, args.chunk_ms / 1000.0)

    return {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_dir": str(args.model_dir),
        "fixtures": fixtures,
        "settings": {
            "chunk_ms": args.chunk_ms,
            "lookahead_ms": args.lookahead_ms,
            "language": args.language,
            "repeats": args.repeats,
            "presets": list(args.presets),
        },
        "environment": {
            "mlx": getattr(mx, "__version__", "unknown"),
            "device": str(mx.default_device()),
            "load_average": list(os.getloadavg()),
        },
        "cases": cases,
    }


def _print_case(case: dict[str, Any], names: list[str], target: float) -> None:
    reference = case["presets"].get("reference")
    ref_p50 = reference["step"]["p50_seconds"] if reference else None
    print(f"\n== {case['seconds']} s ({case['windows']} steps per pass) ==", flush=True)
    header = (
        f"{'preset':<26}{'p50':>9}{'p95':>9}{'mean':>9}{'vs ref p50':>12}"
        f"{'ratio':>8}{'max wait':>10}{'same text':>11}"
    )
    print(header)
    print("-" * len(header))
    for name in names:
        item = case["presets"][name]
        step = item["step"]
        speed = (ref_p50 / step["p50_seconds"]) if ref_p50 else None
        same = item["text_matches_reference"]
        print(
            f"{name:<26}"
            f"{step['p50_seconds'] * 1000:>8.1f}m"
            f"{step['p95_seconds'] * 1000:>8.1f}m"
            f"{step['mean_seconds'] * 1000:>8.1f}m"
            f"{(f'{speed:.2f}x' if speed else 'n/a'):>12}"
            f"{item['queue']['processing_audio_ratio']:>8.3f}"
            f"{item['queue']['max_queue_wait_seconds'] * 1000:>9.1f}m"
            f"{('n/a' if same is None else ('yes' if same else 'NO')):>11}",
            flush=True,
        )
    print(f"(target: p95 <= {target * 1000:.0f} ms)", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bench = _load_benchmark_module()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", default=bench.DEFAULT_MODEL)
    parser.add_argument("--fixtures", nargs="+", default=list(bench.DEFAULT_FIXTURES))
    parser.add_argument("--lengths", type=int, nargs="+", default=list(DEFAULT_LENGTHS))
    parser.add_argument(
        "--presets",
        nargs="+",
        default=[
            "reference",
            "native_linear",
            "native_lm_head",
            "last_logits",
            "fused_attention",
            "audio_bf16",
            "residual_bf16",
            "decoder_native",
            "decoder_native_fp32_head",
            "all",
        ],
    )
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--lookahead-ms", type=int, default=160)
    parser.add_argument("--language", default=None)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    path = Path(
        args.report
        or Path(os.environ.get("TMPDIR", "/tmp")) / "r2t2_compute_ablation.json"
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport: {path}")


if __name__ == "__main__":
    main()
