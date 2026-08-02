#!/usr/bin/env python3
"""Time one real dots.tts request per public variant and output path."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import mlx.core as mx


VARIANTS = ("mf", "soar")
PATHS = ("batch", "stream")
DEFAULT_TEXT = (
    "Technology is most useful when it gives people more time to think, "
    "create, and care for one another."
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def measure_request(
    generator: Any,
    *,
    path: str,
    text: str,
    reference_audio: Path,
    seed: int,
    max_audio_patches: int,
    eos_threshold: float,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Measure one complete request at the public waveform boundary."""

    mx.reset_peak_memory()
    started = clock()
    first_audio_seconds: float | None = None

    if path == "batch":
        output = generator.synthesize(
            text,
            reference_audio=reference_audio,
            seed=seed,
            max_audio_patches=max_audio_patches,
            eos_threshold=eos_threshold,
        )
        waveform_samples = int(output.waveform.size)
        patch_count = int(output.num_patches)
    elif path == "stream":
        waveform_samples = 0
        patch_count = 0
        stream: Iterator[Any] = generator.synthesize_stream(
            text,
            reference_audio=reference_audio,
            seed=seed,
            max_audio_patches=max_audio_patches,
            eos_threshold=eos_threshold,
        )
        try:
            for chunk in stream:
                if first_audio_seconds is None:
                    first_audio_seconds = clock() - started
                waveform_samples += int(chunk.waveform.size)
                patch_count += int(chunk.num_patches)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    else:
        raise ValueError(f"unsupported dots.tts timing path: {path}")

    total_seconds = clock() - started
    if path == "batch":
        first_audio_seconds = total_seconds
    if waveform_samples <= 0 or patch_count <= 0:
        raise RuntimeError("dots.tts timing request produced no complete audio")
    duration_seconds = waveform_samples / int(generator.sample_rate)
    if duration_seconds <= 0:
        raise RuntimeError("dots.tts timing request has invalid waveform duration")
    return {
        "path": path,
        "total_seconds": total_seconds,
        "first_audio_seconds": first_audio_seconds,
        "waveform_samples": waveform_samples,
        "waveform_duration_seconds": duration_seconds,
        "rtf": total_seconds / duration_seconds,
        "patch_count": patch_count,
        "stop_reason": (
            "patch_budget" if patch_count >= max_audio_patches else "eos"
        ),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def compare_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare matching raw cells without an experiment registry or ledger."""

    if before.get("config") != after.get("config"):
        raise ValueError("dots.tts timing configurations differ")
    before_cases = {
        (case["variant"], case["path"]): case for case in before.get("cases", [])
    }
    after_cases = {
        (case["variant"], case["path"]): case for case in after.get("cases", [])
    }
    if not before_cases or set(before_cases) != set(after_cases):
        raise ValueError("dots.tts timing case sets differ")

    cells = []
    for key in sorted(before_cases):
        before_seconds = float(before_cases[key]["total_seconds"])
        after_seconds = float(after_cases[key]["total_seconds"])
        if before_seconds <= 0 or after_seconds <= 0:
            raise ValueError("dots.tts timing totals must be positive")
        cells.append(
            {
                "variant": key[0],
                "path": key[1],
                "before_seconds": before_seconds,
                "after_seconds": after_seconds,
                "improvement": 1.0 - after_seconds / before_seconds,
                "faster": after_seconds < before_seconds,
                "within_two_percent_regression": after_seconds <= before_seconds * 1.02,
            }
        )
    return {
        "cells": cells,
        "passed": all(
            cell["faster"] and cell["within_two_percent_regression"]
            for cell in cells
        ),
    }


def run(
    args: argparse.Namespace,
    *,
    generator_loader: Callable[[Path], Any] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    if generator_loader is None:
        from mlx_speech.generation.dots_tts import DotsTTSGenerator

        generator_loader = DotsTTSGenerator.from_dir

    config = {
        "artifact_class": args.artifact_class,
        "reference_audio": str(args.reference_audio),
        "text": args.text,
        "seed": args.seed,
        "max_audio_patches": args.max_audio_patches,
        "eos_threshold": args.eos_threshold,
        "variants": list(args.variants),
        "paths": list(args.paths),
    }
    cases = []
    for variant in args.variants:
        model_dir = args.model_root / variant / f"mlx-{args.artifact_class}"
        for path in args.paths:
            load_started = clock()
            generator = generator_loader(model_dir)
            load_seconds = clock() - load_started
            case = measure_request(
                generator,
                path=path,
                text=args.text,
                reference_audio=args.reference_audio,
                seed=args.seed,
                max_audio_patches=args.max_audio_patches,
                eos_threshold=args.eos_threshold,
                clock=clock,
            )
            case.update(
                {
                    "variant": variant,
                    "model_dir": str(model_dir),
                    "load_seconds": load_seconds,
                }
            )
            cases.append(case)
            print(
                f"[{variant}/{path}] load={load_seconds:.3f}s "
                f"request={case['total_seconds']:.3f}s "
                f"audio={case['waveform_duration_seconds']:.3f}s "
                f"rtf={case['rtf']:.3f}",
                flush=True,
            )
            del generator
            gc.collect()
            mx.clear_cache()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "config": config,
        "cases": cases,
    }
    if args.compare is not None:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        payload["comparison"] = compare_reports(before, payload)
        payload["passed"] = bool(payload["comparison"]["passed"])
    else:
        payload["passed"] = True
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models/dots_tts"))
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=Path("outputs/source/hank_hill_ref.wav"),
    )
    parser.add_argument("--artifact-class", choices=("base", "bf16", "int8"), default="int8")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=PATHS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-audio-patches", type=int, default=500)
    parser.add_argument("--eos-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    if args.max_audio_patches < 1 or args.max_audio_patches > 512:
        parser.error("--max-audio-patches must be in [1, 512]")
    if not 0.0 <= args.eos_threshold <= 1.0:
        parser.error("--eos-threshold must be in [0, 1]")
    return args


def main() -> None:
    args = parse_args()
    payload = run(args)
    _write_json(args.output, payload)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
