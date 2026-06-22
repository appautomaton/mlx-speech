#!/usr/bin/env python3
"""Quantization gate for Qwen3-ASR: compare bf16 vs int8 vs mxfp8.

Each build is loaded and run STRICTLY SEQUENTIALLY — only one model is ever
resident in memory at a time (load -> transcribe all clips -> free -> next
build). No parallel workers, so it won't choke the machine.

The bf16 transcript is the reference; for each quantized build we report how far
its transcript diverges from bf16 (WER over words, CER over characters — the
latter is the meaningful one for Chinese), plus real-time factor, peak MLX
memory, and on-disk weight size. The relative comparison needs no ground-truth
labels. The JSON report is written under $TMPDIR.

    python scripts/eval/qwen3_asr_quant_gate.py
    python scripts/eval/qwen3_asr_quant_gate.py --audio a.wav b.wav --language Chinese
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_BUILDS: dict[str, str] = {
    "bf16": "models/qwen3_asr_1_7b/mlx-bf16",
    "int8": "models/qwen3_asr_1_7b/mlx-int8",
    "mxfp8": "models/qwen3_asr_1_7b/mlx-mxfp8",
}
DEFAULT_AUDIO: list[str] = [
    "outputs/source/hank_hill_ref.wav",
    "outputs/source/peggy_hill_ref.wav",
]


# --------------------------------------------------------------------------- #
# Metrics — unicode-safe, work for both English (WER) and Chinese (CER)
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).strip().casefold().split())


def _edit_distance(ref: list[str], hyp: list[str]) -> int:
    n, m = len(ref), len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ref_i = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ref_i == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def wer(reference: str, hypothesis: str) -> float:
    r, h = _normalize(reference).split(), _normalize(hypothesis).split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def cer(reference: str, hypothesis: str) -> float:
    r = list(_normalize(reference).replace(" ", ""))
    h = list(_normalize(hypothesis).replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "n/a"
    return f"{n / 1024**3:.2f} GiB"


# --------------------------------------------------------------------------- #
# Per-build run — sequential; one model resident at a time
# --------------------------------------------------------------------------- #
def run_build(name: str, path: str, clips: list[str], *, language: str | None,
              max_new_tokens: int) -> dict:
    import mlx_speech
    from mlx_speech.audio import load_audio
    from mlx_speech.diagnostics import (
        clear_mlx_cache,
        reset_mlx_peak_memory,
        snapshot_mlx_memory,
    )

    weight_file = Path(path) / "model.safetensors"
    size_bytes = weight_file.stat().st_size if weight_file.exists() else None

    reset_mlx_peak_memory()
    asr = mlx_speech.asr.load(path)

    records, total_audio_s, total_infer_s = [], 0.0, 0.0
    for clip in clips:
        samples, _ = load_audio(clip, sample_rate=16000)
        audio = np.asarray(samples, dtype=np.float32)
        dur = len(audio) / 16000.0
        try:
            t0 = time.perf_counter()
            out = asr.generate(audio, sample_rate=16000, language=language,
                               max_new_tokens=max_new_tokens)
            infer = time.perf_counter() - t0
            rec = {"clip": os.path.basename(clip), "duration_s": round(dur, 3),
                   "infer_s": round(infer, 3),
                   "rtf": round(infer / dur, 4) if dur else None,
                   "language": out.language, "text": out.text, "error": None}
            total_audio_s += dur
            total_infer_s += infer
        except Exception as exc:  # noqa: BLE001 — surface, don't abort the sweep
            rec = {"clip": os.path.basename(clip), "duration_s": round(dur, 3),
                   "infer_s": None, "rtf": None, "language": None, "text": None,
                   "error": f"{type(exc).__name__}: {exc}"}
        print(f"  [{name}] {rec['clip']}: "
              + (rec["error"] if rec["error"] else f"rtf={rec['rtf']} "
                 f"text={rec['text'][:80]!r}"))
        records.append(rec)

    peak = snapshot_mlx_memory(f"{name}:after").peak_bytes
    del asr
    gc.collect()
    clear_mlx_cache()

    return {"build": name, "path": path, "size_bytes": size_bytes, "peak_bytes": peak,
            "rtf": round(total_infer_s / total_audio_s, 4) if total_audio_s else None,
            "records": records}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", nargs="+", default=DEFAULT_AUDIO,
                   help="Audio clips (the same set runs through every build).")
    p.add_argument("--language", default=None,
                   help="Force a language (default: auto-detect).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--report", default=None,
                   help="JSON report path (default: $TMPDIR/qwen3_asr_quant_gate.json).")
    p.add_argument("--builds", nargs="+", default=None,
                   help="Override builds as name=path pairs; first is the reference.")
    return p.parse_args()


def _resolve_builds(arg: list[str] | None) -> dict[str, str]:
    if not arg:
        return dict(DEFAULT_BUILDS)
    out: dict[str, str] = {}
    for item in arg:
        name, _, path = item.partition("=")
        out[name] = path or name
    return out


def main() -> None:
    args = parse_args()
    builds = _resolve_builds(args.builds)
    clips = [c for c in args.audio if Path(c).exists()]
    missing = [c for c in args.audio if not Path(c).exists()]
    for c in missing:
        print(f"WARN: skipping missing clip {c}")
    if not clips:
        raise SystemExit("No audio clips found.")

    print(f"Clips: {len(clips)} | builds: {', '.join(builds)} | "
          f"language: {args.language or 'auto'}\n")

    results = []
    for name, path in builds.items():
        print(f"=== build {name} ({path}) ===")
        results.append(run_build(name, path, clips, language=args.language,
                                 max_new_tokens=args.max_new_tokens))
        print()

    # Reference = first build (bf16). Compute divergence per build.
    ref_by_clip = {r["clip"]: r for r in results[0]["records"]}
    for res in results:
        wers, cers, exact = [], [], 0
        for rec in res["records"]:
            ref = ref_by_clip.get(rec["clip"])
            if not ref or ref.get("text") is None or rec.get("text") is None:
                continue
            w, c = wer(ref["text"], rec["text"]), cer(ref["text"], rec["text"])
            wers.append(w)
            cers.append(c)
            exact += int(_normalize(ref["text"]) == _normalize(rec["text"]))
        res["mean_wer_vs_ref"] = round(sum(wers) / len(wers), 4) if wers else None
        res["mean_cer_vs_ref"] = round(sum(cers) / len(cers), 4) if cers else None
        res["exact_match"] = f"{exact}/{len(res['records'])}"

    ref_name = results[0]["build"]
    print(f"\n{'='*78}\nQUANTIZATION GATE — divergence vs {ref_name} reference\n{'='*78}")
    header = f"{'build':<8}{'size':>11}{'peak_mem':>12}{'RTF':>9}{'WER':>9}{'CER':>9}{'exact':>9}"
    print(header)
    print("-" * len(header))
    for res in results:
        wer_s = "ref" if res["build"] == ref_name else (
            f"{res['mean_wer_vs_ref']:.4f}" if res["mean_wer_vs_ref"] is not None else "n/a")
        cer_s = "ref" if res["build"] == ref_name else (
            f"{res['mean_cer_vs_ref']:.4f}" if res["mean_cer_vs_ref"] is not None else "n/a")
        rtf_s = f"{res['rtf']:.4f}" if res["rtf"] is not None else "n/a"
        print(f"{res['build']:<8}{_fmt_bytes(res['size_bytes']):>11}"
              f"{_fmt_bytes(res['peak_bytes']):>12}{rtf_s:>9}{wer_s:>9}{cer_s:>9}"
              f"{res['exact_match']:>9}")
    print("\nLower WER/CER vs bf16 = more faithful quantization (0 = identical).")
    print("RTF = infer_time / audio_duration (lower is faster).")

    report_path = Path(args.report or (
        Path(os.environ.get("TMPDIR", "/tmp")) / "qwen3_asr_quant_gate.json"))
    report_path.write_text(json.dumps({"builds": results}, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
