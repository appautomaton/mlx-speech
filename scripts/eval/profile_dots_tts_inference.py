#!/usr/bin/env python3
"""Profile the local cached dots.tts inference paths on Apple Silicon."""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    from scripts.eval.dots_tts_comparison_contract import (
        load_comparison_contract,
        update_comparison_contract,
    )
except ModuleNotFoundError:  # Direct ``python scripts/eval/...`` execution.
    from dots_tts_comparison_contract import (  # type: ignore[no-redef]
        load_comparison_contract,
        update_comparison_contract,
    )


VARIANTS = ("mf", "soar")
PATHS = ("batch", "stream")
WEIGHT_FILES = (
    "core.safetensors",
    "vocoder.safetensors",
    "speaker.safetensors",
    "latent_stats.safetensors",
)
DEFAULT_TEXT = (
    "Technology is most useful when it gives people more time to think, "
    "create, and care for one another."
)
SOURCE_PATHS = ("src", "scripts", "tests", "pyproject.toml", "uv.lock")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(*args: str) -> str:
    result = subprocess.run(
        ("git", *args),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="surrogateescape").strip()


def _source_identity() -> dict[str, Any]:
    digest = hashlib.sha256()
    raw_paths = subprocess.run(
        (
            "git",
            "ls-files",
            "-co",
            "--exclude-standard",
            "-z",
            "--",
            *SOURCE_PATHS,
        ),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    files = []
    for raw_path in raw_paths.split(b"\0"):
        if not raw_path:
            continue
        path = Path(raw_path.decode("utf-8", errors="surrogateescape"))
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        path_bytes = str(path).encode("utf-8", errors="surrogateescape")
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        files.append(str(path))
    diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD", "--", *SOURCE_PATHS),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("symbolic-ref", "--short", "HEAD"),
        "source_tree_sha256": digest.hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "file_count": len(files),
        "paths": list(SOURCE_PATHS),
    }


def _artifact_inventory(path: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in (*WEIGHT_FILES, "mlx_config.json"):
        file = path / name
        if not file.is_file():
            raise FileNotFoundError(f"dots.tts artifact file is missing: {file}")
        files[name] = {"bytes": file.stat().st_size, "sha256": _sha256(file)}
    metadata = json.loads((path / "mlx_config.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {
        "path": str(path),
        "artifact_class": metadata["artifact_class"],
        "source": metadata["source"],
        "quantization": metadata["quantization"],
        "digest": digest,
        "files": files,
    }


def _host_identity() -> dict[str, str]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }


class _StageProfiler:
    def __init__(self, generator: Any):
        self.generator = generator
        self.seconds: dict[str, float] = {}
        self._originals: list[tuple[Any, str, Any]] = []

    def _wrap(self, target: Any, name: str, stage: str) -> None:
        if not hasattr(target, name):
            return
        original = getattr(target, name)

        @functools.wraps(original)
        def measured(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.seconds[stage] = self.seconds.get(stage, 0.0) + (
                    time.perf_counter() - started
                )

        self._originals.append((target, name, original))
        setattr(target, name, measured)

    def __enter__(self) -> "_StageProfiler":
        self._wrap(self.generator, "prepare_prompt", "prompt")
        self._wrap(self.generator, "_prefill", "prefill")
        self._wrap(self.generator, "_solve_patch", "acoustic")
        self._wrap(self.generator, "_decode_request_chunk", "decoder")
        self._wrap(self.generator.components.audio_vae, "decode", "decoder")
        return self

    def __exit__(self, *_args: object) -> None:
        for target, name, original in reversed(self._originals):
            setattr(target, name, original)
        self._originals.clear()

    def reset(self) -> None:
        self.seconds.clear()

    def result(self, total_seconds: float) -> dict[str, float]:
        recorded = {name: float(value) for name, value in sorted(self.seconds.items())}
        recorded["residual"] = max(0.0, total_seconds - sum(recorded.values()))
        return recorded


def _clear_prompt_cache(generator: Any) -> bool:
    clear = getattr(generator, "clear_prompt_cache", None)
    if callable(clear):
        clear()
        return True
    cache = getattr(generator, "_prompt_cache", None)
    if cache is not None and hasattr(cache, "clear"):
        lock = getattr(generator, "_prompt_cache_lock", None)
        if lock is None:
            cache.clear()
        else:
            with lock:
                cache.clear()
        return True
    return False


def _health(waveform: Any) -> dict[str, Any]:
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    return {
        "finite": bool(np.all(np.isfinite(values))),
        "non_silent": bool(np.any(np.abs(values) > 0.0)),
        "peak_absolute": float(np.max(np.abs(values), initial=0.0)),
    }


def measure_trial(
    generator: Any,
    profiler: _StageProfiler,
    *,
    path: str,
    text: str,
    reference_audio: Path,
    seed: int,
    max_audio_patches: int,
    eos_threshold: float,
    memory_limit_bytes: int,
) -> dict[str, Any]:
    """Measure one request without clearing model or compilation caches."""

    import mlx.core as mx

    if path not in PATHS:
        raise ValueError(f"unsupported dots.tts profile path: {path}")
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    profiler.reset()
    started = time.perf_counter()
    first_output_seconds: float | None = None
    patch_count = 0
    chunk_count = 0
    waveforms: list[Any] = []
    if path == "batch":
        result = generator.synthesize(
            text,
            reference_audio=reference_audio,
            max_audio_patches=max_audio_patches,
            seed=seed,
            eos_threshold=eos_threshold,
        )
        mx.eval(result.waveform)
        first_output_seconds = time.perf_counter() - started
        patch_count = int(result.num_patches)
        chunk_count = 1
        waveforms.append(result.waveform)
    else:
        stream: Iterator[Any] = generator.synthesize_stream(
            text,
            reference_audio=reference_audio,
            max_audio_patches=max_audio_patches,
            seed=seed,
            eos_threshold=eos_threshold,
        )
        try:
            for chunk in stream:
                mx.eval(chunk.waveform)
                if first_output_seconds is None:
                    first_output_seconds = time.perf_counter() - started
                waveforms.append(chunk.waveform)
                patch_count += int(chunk.num_patches)
                chunk_count += 1
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()
    total_seconds = time.perf_counter() - started
    peak = int(mx.get_peak_memory())
    if first_output_seconds is None or not waveforms:
        raise RuntimeError(f"dots.tts {path} profile produced no waveform")
    if patch_count != max_audio_patches:
        raise RuntimeError(
            f"dots.tts {path} profile produced {patch_count} payload patches; "
            f"expected {max_audio_patches}"
        )
    if peak >= memory_limit_bytes:
        raise MemoryError(
            f"dots.tts {path} peak {peak} must remain below {memory_limit_bytes}"
        )
    waveform = waveforms[0] if len(waveforms) == 1 else mx.concatenate(waveforms)
    output = _health(waveform)
    if not output["finite"] or not output["non_silent"]:
        raise RuntimeError(f"dots.tts {path} profile produced unhealthy waveform")
    waveform_samples = int(waveform.size)
    output_seconds = waveform_samples / float(generator.sample_rate)
    return {
        "path": path,
        "seed": seed,
        "patch_count": patch_count,
        "chunk_count": chunk_count,
        "waveform_samples": waveform_samples,
        "sample_rate": int(generator.sample_rate),
        "output_seconds": output_seconds,
        "total_seconds": total_seconds,
        "first_output_seconds": first_output_seconds,
        "completion_after_first_output_seconds": total_seconds - first_output_seconds,
        "rtf": total_seconds / output_seconds,
        "stage_seconds": profiler.result(total_seconds),
        "baseline_memory_bytes": baseline,
        "peak_memory_bytes": peak,
        "incremental_peak_bytes": max(0, peak - baseline),
        "output_health": output,
    }


def summarize_case(
    variant: str,
    path: str,
    warmup: dict[str, Any],
    trials: list[dict[str, Any]],
) -> dict[str, Any]:
    if not trials:
        raise ValueError("dots.tts profile case requires measured trials")
    stage_names = sorted(
        {name for trial in trials for name in trial["stage_seconds"]}
    )
    return {
        "variant": variant,
        "path": path,
        "warmup": warmup,
        "medians": {
            "total_seconds": statistics.median(
                float(trial["total_seconds"]) for trial in trials
            ),
            "first_output_seconds": statistics.median(
                float(trial["first_output_seconds"]) for trial in trials
            ),
            "output_seconds": statistics.median(
                float(trial["output_seconds"]) for trial in trials
            ),
            "rtf": statistics.median(float(trial["rtf"]) for trial in trials),
            "peak_memory_bytes": max(
                int(trial["peak_memory_bytes"]) for trial in trials
            ),
            "stage_seconds": {
                name: statistics.median(
                    float(trial["stage_seconds"].get(name, 0.0))
                    for trial in trials
                )
                for name in stage_names
            },
        },
        "trials": trials,
    }


def _maximum_budget_smoke(
    generator: Any,
    *,
    text: str,
    reference_audio: Path,
    seed: int,
    patch_count: int,
    memory_limit_bytes: int,
) -> dict[str, Any]:
    import mlx.core as mx

    states: list[Any] = []
    original = generator._new_dit_request

    @functools.wraps(original)
    def capture(max_audio_patches: int) -> tuple[Any, Any]:
        solver, state = original(max_audio_patches)
        states.append(state)
        return solver, state

    generator._new_dit_request = capture
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    emitted = 0
    waveform_samples = 0
    stream = generator.synthesize_stream(
        text,
        reference_audio=reference_audio,
        max_audio_patches=512,
        seed=seed,
        eos_threshold=1.0,
    )
    try:
        for chunk in stream:
            mx.eval(chunk.waveform)
            emitted += int(chunk.num_patches)
            waveform_samples += int(chunk.waveform.size)
            if emitted >= patch_count:
                break
    finally:
        close = getattr(stream, "close", None)
        if callable(close):
            close()
        generator._new_dit_request = original
    peak = int(mx.get_peak_memory())
    if emitted != patch_count:
        raise RuntimeError(
            f"dots.tts maximum-budget smoke emitted {emitted}; expected {patch_count}"
        )
    if not states or states[-1] is None or states[-1].cache is None:
        raise RuntimeError("dots.tts maximum-budget smoke did not publish DiT cache")
    state = states[-1]
    cache = state.cache
    return {
        "request_capacity_patches": int(state.capacity_patches),
        "physical_capacity_patches": int(cache.capacity_patches),
        "published_tokens": [int(offset) for offset in cache.offsets],
        "emitted_patches": emitted,
        "waveform_samples": waveform_samples,
        "baseline_memory_bytes": baseline,
        "peak_memory_bytes": peak,
        "incremental_peak_bytes": max(0, peak - baseline),
        "below_memory_limit": peak < memory_limit_bytes,
    }


def _case_map(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(case["variant"]), str(case["path"])): case
        for case in payload["cases"]
    }


def compare_performance(
    current: dict[str, Any],
    contract_path: Path,
    *,
    minimum_batch_improvement: float,
) -> dict[str, Any]:
    contract = load_comparison_contract(contract_path)
    evidence = contract["performance"]
    baseline = evidence.get("baseline")
    if not isinstance(baseline, dict):
        raise ValueError("dots.tts performance contract is missing its baseline")
    if current["host"] != baseline.get("host"):
        raise ValueError("dots.tts performance host differs from baseline")
    if current["reference"]["sha256"] != baseline.get("reference", {}).get("sha256"):
        raise ValueError("dots.tts performance reference differs from baseline")
    fixed_fields = (
        "artifact_class",
        "text",
        "seed",
        "max_audio_patches",
        "eos_threshold",
        "warmup_runs",
        "runs",
        "variants",
        "paths",
    )
    for field in fixed_fields:
        if current["config"].get(field) != baseline.get("config", {}).get(field):
            raise ValueError(f"dots.tts performance workload differs at {field}")
    for variant in current["config"]["variants"]:
        current_artifact = current["artifacts"][variant]["digest"]
        baseline_artifact = baseline.get("artifacts", {}).get(variant, {}).get("digest")
        if current_artifact != baseline_artifact:
            raise ValueError(f"dots.tts {variant} artifact differs from baseline")
    baseline_cases = _case_map(baseline)
    current_cases = _case_map(current)
    if set(current_cases) != set(baseline_cases):
        raise ValueError("dots.tts performance case sets differ from baseline")
    variants = {}
    for variant in current["config"]["variants"]:
        before = float(baseline_cases[(variant, "batch")]["medians"]["total_seconds"])
        after = float(current_cases[(variant, "batch")]["medians"]["total_seconds"])
        improvement = 1.0 - after / before
        variants[variant] = {
            "baseline_batch_seconds": before,
            "current_batch_seconds": after,
            "improvement": improvement,
            "passed": improvement >= minimum_batch_improvement,
        }
    return {
        "contract": str(contract_path),
        "contract_performance_report_sha256": evidence.get("report_sha256"),
        "minimum_batch_improvement": minimum_batch_improvement,
        "variants": variants,
        "passed": all(bool(item["passed"]) for item in variants.values()),
    }


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_speech.generation.dots_tts import DotsTTSGenerator

    memory_limit_bytes = round(args.memory_limit_gib * 1024**3)
    mx.set_memory_limit(memory_limit_bytes)
    mx.set_cache_limit(2 * 1024**3)
    source = _source_identity()
    reference = {
        "path": str(args.reference_audio),
        "bytes": args.reference_audio.stat().st_size,
        "sha256": _sha256(args.reference_audio),
    }
    artifacts = {
        variant: _artifact_inventory(
            args.model_root / variant / f"mlx-{args.artifact_class}"
        )
        for variant in args.variants
    }
    cases = []
    smokes = []
    for variant in args.variants:
        model_dir = args.model_root / variant / f"mlx-{args.artifact_class}"
        generator = DotsTTSGenerator.from_dir(model_dir)
        with _StageProfiler(generator) as stage_profiler:
            for path in args.paths:
                warmup_trials = []
                for warmup_index in range(args.warmup_runs):
                    trial = measure_trial(
                        generator,
                        stage_profiler,
                        path=path,
                        text=args.text,
                        reference_audio=args.reference_audio,
                        seed=args.seed,
                        max_audio_patches=args.max_audio_patches,
                        eos_threshold=args.eos_threshold,
                        memory_limit_bytes=memory_limit_bytes,
                    )
                    trial["run"] = warmup_index + 1
                    warmup_trials.append(trial)
                    print(
                        f"[{variant}/{path} warmup {warmup_index + 1}] "
                        f"total={trial['total_seconds']:.3f}s",
                        flush=True,
                    )
                prompt_cache_cleared = _clear_prompt_cache(generator)
                trials = []
                for run_index in range(args.runs):
                    trial = measure_trial(
                        generator,
                        stage_profiler,
                        path=path,
                        text=args.text,
                        reference_audio=args.reference_audio,
                        seed=args.seed,
                        max_audio_patches=args.max_audio_patches,
                        eos_threshold=args.eos_threshold,
                        memory_limit_bytes=memory_limit_bytes,
                    )
                    trial["run"] = run_index + 1
                    trial["reference_cache"] = (
                        "cold" if run_index == 0 else "warm"
                    )
                    trials.append(trial)
                    print(
                        f"[{variant}/{path} run {run_index + 1}] "
                        f"total={trial['total_seconds']:.3f}s "
                        f"first={trial['first_output_seconds']:.3f}s "
                        f"peak={trial['peak_memory_bytes'] / 1024**3:.3f}GiB",
                        flush=True,
                    )
                warmup = {
                    "runs": warmup_trials,
                    "total_seconds": sum(
                        float(item["total_seconds"]) for item in warmup_trials
                    ),
                    "explicit_compile_seconds": 0.0,
                    "prompt_cache_cleared_after": prompt_cache_cleared,
                }
                cases.append(summarize_case(variant, path, warmup, trials))
        if args.maximum_bucket_smoke_patches:
            smoke = _maximum_budget_smoke(
                generator,
                text=args.text,
                reference_audio=args.reference_audio,
                seed=args.seed,
                patch_count=args.maximum_bucket_smoke_patches,
                memory_limit_bytes=memory_limit_bytes,
            )
            smoke["variant"] = variant
            smokes.append(smoke)
        del generator
        gc.collect()
        mx.clear_cache()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "host": _host_identity(),
        "mlx_version": importlib.metadata.version("mlx"),
        "source": source,
        "command": " ".join(sys.argv),
        "config": {
            "model_root": str(args.model_root),
            "artifact_class": args.artifact_class,
            "text": args.text,
            "seed": args.seed,
            "max_audio_patches": args.max_audio_patches,
            "eos_threshold": args.eos_threshold,
            "warmup_runs": args.warmup_runs,
            "runs": args.runs,
            "variants": list(args.variants),
            "paths": list(args.paths),
            "memory_limit_gib": args.memory_limit_gib,
            "solver_steps": "artifact_default",
        },
        "reference": reference,
        "artifacts": artifacts,
        "cases": cases,
        "maximum_budget_smokes": smokes,
        "passed": all(
            bool(trial["output_health"]["finite"])
            and bool(trial["output_health"]["non_silent"])
            and int(trial["peak_memory_bytes"]) < memory_limit_bytes
            for case in cases
            for trial in case["trials"]
        )
        and all(bool(smoke["below_memory_limit"]) for smoke in smokes),
    }
    if args.comparison_contract is not None:
        payload["comparison"] = compare_performance(
            payload,
            args.comparison_contract,
            minimum_batch_improvement=args.minimum_batch_improvement,
        )
        payload["passed"] = bool(payload["passed"]) and bool(
            payload["comparison"]["passed"]
        )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=Path("models/dots_tts"))
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=Path("outputs/source/hank_hill_ref.wav"),
    )
    parser.add_argument("--artifact-class", choices=("base", "int8"), default="base")
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=PATHS)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-audio-patches", type=int, default=128)
    parser.add_argument("--eos-threshold", type=float, default=1.0)
    parser.add_argument("--memory-limit-gib", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-comparison-contract", type=Path)
    parser.add_argument("--comparison-contract", type=Path)
    parser.add_argument("--minimum-batch-improvement", type=float, default=0.0)
    parser.add_argument("--maximum-bucket-smoke-patches", type=int, default=0)
    args = parser.parse_args()
    if args.warmup_runs < 1 or args.runs < 1:
        parser.error("warmup and measured run counts must be positive")
    if args.max_audio_patches < 1 or args.max_audio_patches > 512:
        parser.error("--max-audio-patches must be in [1, 512]")
    if args.memory_limit_gib <= 0:
        parser.error("--memory-limit-gib must be positive")
    if args.minimum_batch_improvement < 0 or args.minimum_batch_improvement >= 1:
        parser.error("--minimum-batch-improvement must be in [0, 1)")
    if args.maximum_bucket_smoke_patches < 0:
        parser.error("--maximum-bucket-smoke-patches must be non-negative")
    if args.freeze_comparison_contract and args.comparison_contract:
        parser.error("freeze and comparison contract modes are mutually exclusive")
    if args.freeze_comparison_contract and (
        args.warmup_runs != 1
        or args.runs != 3
        or args.max_audio_patches != 128
        or args.seed != 42
        or args.eos_threshold != 1.0
        or tuple(args.variants) != VARIANTS
        or tuple(args.paths) != PATHS
        or args.artifact_class != "base"
    ):
        parser.error("frozen baseline must use the approved fixed workload")
    if args.comparison_contract and args.minimum_batch_improvement <= 0:
        parser.error("comparison mode requires --minimum-batch-improvement")
    if args.minimum_batch_improvement and args.comparison_contract is None:
        parser.error("improvement gate requires --comparison-contract")
    return args


def main() -> None:
    args = parse_args()
    payload = profile(args)
    _write_json(args.output, payload)
    if args.freeze_comparison_contract is not None:
        update_comparison_contract(
            args.freeze_comparison_contract,
            section="performance",
            evidence={
                "report_sha256": _sha256(args.output),
                "baseline": payload,
            },
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not payload["passed"]:
        raise SystemExit("dots.tts cached inference profile failed")


if __name__ == "__main__":
    main()
