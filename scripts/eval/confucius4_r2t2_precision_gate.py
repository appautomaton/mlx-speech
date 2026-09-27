#!/usr/bin/env python3
"""Phase 2 precision gate for Confucius4-R2T2 streaming compute options.

Compares a candidate step-compute preset with the reference (float32-cast)
path in two ways:

* Teacher-forced windows. A reference streaming session records every step
  call (prefix, audio, budget). Each call is replayed under the reference and
  the candidate with identical inputs. Per greedy choice the gate records the
  largest last-position logit difference over the vocabulary, the difference
  at the reference's top-2 tokens, and the reference top-1/top-2 margin. The
  first differing generated token is a numerical tie only when the reference
  margin there is below tau.
* Free-running sessions. Reference and candidate sessions run end to end; the
  gate reports relative CER of the candidate's final transcript.

``calibrate`` derives tau and the logit tolerance from calibration fixtures.
``gate`` applies fixed values, which must be recorded before the run.

    .venv/bin/python scripts/eval/confucius4_r2t2_precision_gate.py calibrate \\
        --candidate all
    .venv/bin/python scripts/eval/confucius4_r2t2_precision_gate.py gate \\
        --candidate all --tau 0.5 --logit-tol 1.0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

SAMPLE_RATE = 16_000
CALIBRATION_FIXTURES = ("models/netease/confucius4_r2t2/samples/test.wav",)
GATE_FIXTURES = (
    "outputs/source/hank_hill_ref.wav",
    "outputs/source/peggy_hill_ref.wav",
    "outputs/source/donald_trump_ref.wav",
)
GATE_LONG_SECONDS = (30, 60)
CER_LIMIT = 0.010


def _load_sibling(name: str):
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(f"_r2t2_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Text metrics
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().casefold().split())


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i] + [0] * len(hypothesis)
        for j, hyp_item in enumerate(hypothesis, start=1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ref_item != hyp_item),
            )
        previous = current
    return previous[-1]


def char_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref = list(normalize_text(reference).replace(" ", ""))
    hyp = list(normalize_text(hypothesis).replace(" ", ""))
    return edit_distance(ref, hyp), len(ref)


# --------------------------------------------------------------------------- #
# Step replay
# --------------------------------------------------------------------------- #
class StepRecord:
    """Greedy trace of one step call under one compute preset."""

    def __init__(self) -> None:
        self.logits: list[Any] = []

    def observe(self, index: int, logits) -> None:
        del index
        self.logits.append(logits)


def compare_step(reference_ids, candidate_ids, reference_logits, candidate_logits):
    """Per greedy choice up to (and including) the first differing token."""

    import mlx.core as mx

    choices = []
    compared = min(len(reference_logits), len(candidate_logits))
    first_mismatch = None
    for index in range(compared):
        ref = reference_logits[index].astype(mx.float32)
        cand = candidate_logits[index].astype(mx.float32)
        top = mx.argpartition(-ref, kth=1)[:2]
        top_values = ref[top]
        order = mx.argsort(-top_values)
        top = top[order]
        diff = mx.abs(ref - cand)
        mx.eval(top, diff)
        top1, top2 = int(top[0].item()), int(top[1].item())
        record = {
            "index": index,
            "ref_top1": top1,
            "cand_top1": int(mx.argmax(cand).item()),
            "margin": float((ref[top1] - ref[top2]).item()),
            "max_abs_diff": float(mx.max(diff).item()),
            "top2_abs_diff": float(max(diff[top1].item(), diff[top2].item())),
        }
        choices.append(record)
        if record["ref_top1"] != record["cand_top1"]:
            first_mismatch = index
            break
    ids_equal = list(reference_ids) == list(candidate_ids)
    return {
        "choices": choices,
        "ids_equal": ids_equal,
        "first_mismatch": first_mismatch,
    }


def record_reference_calls(adapter, audio, *, language, chunk_ms, lookahead_ms):
    """Run a reference session and return every step call plus the final text."""

    from mlx_speech.asr._adapters.confucius4_r2t2 import (
        REFERENCE_STREAM_COMPUTE,
        _completion,
    )
    from mlx_speech.models.confucius4_r2t2.streaming import R2T2StreamSession

    runtime = adapter._runtime
    tokenizer = runtime.processor.tokenizer
    calls: list[dict[str, Any]] = []

    def decoder(prefix: str, pcm: np.ndarray, max_new_tokens: int) -> str:
        calls.append(
            {"prefix": prefix, "samples": int(pcm.shape[0]), "budget": int(max_new_tokens)}
        )
        return _completion(
            runtime,
            prompt_suffix=prefix,
            audio=pcm,
            max_new_tokens=max_new_tokens,
            context="",
            language=language,
            compute=REFERENCE_STREAM_COMPUTE,
        )

    session = R2T2StreamSession(
        decoder,
        tokenizer.encode,
        lambda ids: tokenizer.decode(list(ids)),
        language=language,
        chunk_ms=chunk_ms,
        lookahead_ms=lookahead_ms,
    )
    step = int(round(chunk_ms / 1000 * SAMPLE_RATE))
    for start in range(0, int(audio.shape[0]), step):
        session.feed(audio[start : start + step])
    return calls, session.finalize().text


def replay(adapter, audio, calls, candidate, *, language):
    from mlx_speech.asr._adapters.confucius4_r2t2 import (
        REFERENCE_STREAM_COMPUTE,
        _completion,
    )

    runtime = adapter._runtime
    results = []
    for number, call in enumerate(calls):
        pcm = audio[: call["samples"]]
        traces = {}
        ids = {}
        for label, compute in (("reference", REFERENCE_STREAM_COMPUTE), ("candidate", candidate)):
            record = StepRecord()
            captured: list[int] = []

            def observer(index, logits, record=record):
                record.observe(index, logits)

            text = _completion(
                runtime,
                prompt_suffix=call["prefix"],
                audio=pcm,
                max_new_tokens=call["budget"],
                context="",
                language=language,
                compute=compute,
                logits_observer=observer,
            )
            traces[label] = record
            ids[label] = (text, captured)
        comparison = compare_step(
            [], [], traces["reference"].logits, traces["candidate"].logits
        )
        comparison["ids_equal"] = ids["reference"][0] == ids["candidate"][0]
        comparison["call"] = number
        comparison["samples"] = call["samples"]
        results.append(comparison)
    return results


# --------------------------------------------------------------------------- #
# Main modes
# --------------------------------------------------------------------------- #
def load_inputs(bench, paths, long_seconds, all_clips):
    from mlx_speech.audio import load_audio

    inputs = []
    for path in paths:
        samples, _ = load_audio(path, sample_rate=SAMPLE_RATE, mono=True)
        clip = np.asarray(samples, dtype=np.float32).reshape(-1)
        inputs.append(
            {"name": Path(path).name, "sha256": bench.sha256_file(Path(path)), "audio": clip}
        )
    for seconds in long_seconds:
        inputs.append(
            {
                "name": f"concat_{seconds}s",
                "sha256": None,
                "audio": bench.build_concatenated_audio(
                    all_clips, target_samples=seconds * SAMPLE_RATE
                ),
            }
        )
    return inputs


def run(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    import mlx_speech
    from mlx_speech.audio import load_audio

    bench = _load_sibling("benchmark_confucius4_r2t2_streaming.py")
    ablation = _load_sibling("confucius4_r2t2_compute_ablation.py")
    presets = ablation.preset_table()
    if args.candidate not in presets:
        raise SystemExit(f"unknown candidate {args.candidate}; known: {', '.join(presets)}")
    candidate = presets[args.candidate]

    all_clips = []
    for path in bench.DEFAULT_FIXTURES:
        samples, _ = load_audio(path, sample_rate=SAMPLE_RATE, mono=True)
        all_clips.append(np.asarray(samples, dtype=np.float32).reshape(-1))

    if args.mode == "calibrate":
        inputs = load_inputs(bench, args.fixtures or CALIBRATION_FIXTURES, (), all_clips)
    else:
        if args.tau is None or args.logit_tol is None:
            raise SystemExit("gate mode needs the recorded --tau and --logit-tol")
        inputs = load_inputs(
            bench, args.fixtures or GATE_FIXTURES, tuple(args.long_seconds), all_clips
        )

    adapter = mlx_speech.asr.load(str(args.model_dir))
    rows = []
    cer_errors = 0
    cer_chars = 0
    for item in inputs:
        audio = item["audio"]
        started = time.perf_counter()
        calls, reference_text = record_reference_calls(
            adapter,
            audio,
            language=args.language,
            chunk_ms=args.chunk_ms,
            lookahead_ms=args.lookahead_ms,
        )
        steps = replay(adapter, audio, calls, candidate, language=args.language)
        session = adapter.stream_session(
            chunk_ms=args.chunk_ms,
            lookahead_ms=args.lookahead_ms,
            language=args.language,
            compute=candidate,
        )
        stride = int(round(args.chunk_ms / 1000 * SAMPLE_RATE))
        for start in range(0, int(audio.shape[0]), stride):
            session.feed(audio[start : start + stride])
        candidate_text = session.finalize().text
        errors, chars = char_errors(reference_text, candidate_text)
        cer_errors += errors
        cer_chars += chars
        choices = [choice for step in steps for choice in step["choices"]]
        mismatches = [step for step in steps if step["first_mismatch"] is not None]
        row = {
            "name": item["name"],
            "sha256": item["sha256"],
            "seconds": round(float(audio.shape[0]) / SAMPLE_RATE, 3),
            "calls": len(calls),
            "choices": len(choices),
            "max_abs_diff": max(choice["max_abs_diff"] for choice in choices),
            "max_top2_abs_diff": max(choice["top2_abs_diff"] for choice in choices),
            "min_margin": min(choice["margin"] for choice in choices),
            "mismatch_steps": [
                {
                    "call": step["call"],
                    "samples": step["samples"],
                    "choice": step["first_mismatch"],
                    "margin": step["choices"][step["first_mismatch"]]["margin"],
                }
                for step in mismatches
            ],
            "reference_text": reference_text,
            "candidate_text": candidate_text,
            "char_errors": errors,
            "reference_chars": chars,
            "wall_seconds": round(time.perf_counter() - started, 1),
        }
        rows.append(row)
        print(
            f"{row['name']:<24} calls={row['calls']:>4} choices={row['choices']:>5} "
            f"max|d|={row['max_abs_diff']:.4f} top2|d|={row['max_top2_abs_diff']:.4f} "
            f"mismatches={len(row['mismatch_steps'])} cer={errors}/{chars} "
            f"[{row['wall_seconds']}s]",
            flush=True,
        )

    report: dict[str, Any] = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": args.mode,
        "candidate": args.candidate,
        "model_dir": str(args.model_dir),
        "settings": {
            "chunk_ms": args.chunk_ms,
            "lookahead_ms": args.lookahead_ms,
            "language": args.language,
        },
        "environment": {"mlx": getattr(mx, "__version__", "unknown")},
        "rows": rows,
    }
    if args.mode == "calibrate":
        top2 = max(row["max_top2_abs_diff"] for row in rows)
        full = max(row["max_abs_diff"] for row in rows)
        report["calibration"] = {
            "observed_max_top2_abs_diff": top2,
            "observed_max_abs_diff": full,
            "tau": 2.0 * top2,
            "logit_tol": 1.5 * full,
            "rule": "tau = 2 x max top-2 |diff|; logit_tol = 1.5 x max |diff|",
        }
        print(json.dumps(report["calibration"], indent=2))
    else:
        failures = []
        for row in rows:
            if row["max_abs_diff"] > args.logit_tol:
                failures.append(f"{row['name']}: max|d| {row['max_abs_diff']:.4f} > tol")
            for mismatch in row["mismatch_steps"]:
                if mismatch["margin"] >= args.tau:
                    failures.append(
                        f"{row['name']}: call {mismatch['call']} diverged with margin "
                        f"{mismatch['margin']:.4f} >= tau"
                    )
        cer = (cer_errors / cer_chars) if cer_chars else 0.0
        if cer > CER_LIMIT:
            failures.append(f"aggregate relative CER {cer:.4f} > {CER_LIMIT}")
        report["gate"] = {
            "tau": args.tau,
            "logit_tol": args.logit_tol,
            "cer_limit": CER_LIMIT,
            "aggregate_relative_cer": cer,
            "failures": failures,
            "passed": not failures,
        }
        print(json.dumps(report["gate"], indent=2, ensure_ascii=False))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bench = _load_sibling("benchmark_confucius4_r2t2_streaming.py")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("calibrate", "gate"))
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--model-dir", default=bench.DEFAULT_MODEL)
    parser.add_argument("--fixtures", nargs="+", default=None)
    parser.add_argument("--long-seconds", type=int, nargs="*", default=list(GATE_LONG_SECONDS))
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--logit-tol", type=float, default=None)
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--lookahead-ms", type=int, default=160)
    parser.add_argument("--language", default=None)
    parser.add_argument("--report", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = run(args)
    path = Path(
        args.report
        or Path(os.environ.get("TMPDIR", "/tmp"))
        / f"r2t2_precision_{args.mode}_{args.candidate}.json"
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report: {path}")
    if args.mode == "gate" and not report["gate"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
