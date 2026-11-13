"""Measure Breeze batch throughput through waveform output on local weights."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import threading
import time
import warnings
from pathlib import Path

import mlx.core as mx

import mlx_speech
from mlx_speech.audio import load_audio, write_wav
from mlx_speech.generation.breeze_tts_batch import generate_breeze_batch

LINES = {
    "zh": [
        "今天的天气很好，我们一起出去走走吧。",
        "请把窗户关上，外面开始下雨了。",
        "别着急，我们还有足够的时间。",
    ],
    "en": [
        "The weather is lovely today. Let's go for a walk.",
        "Please close the window. It has started raining outside.",
        "Take your time. We still have plenty of time left.",
    ],
}


class _ProcessUsage(ctypes.Structure):
    """macOS rusage_info_v4, including allocator-independent footprint peaks."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size "
            "resident_size phys_footprint proc_start_abstime proc_exit_abstime "
            "child_user_time child_system_time child_pkg_idle_wkups child_interrupt_wkups "
            "child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
            "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background "
            "cpu_time_qos_utility cpu_time_qos_legacy cpu_time_qos_user_initiated "
            "cpu_time_qos_user_interactive billed_system_time serviced_system_time "
            "logical_writes lifetime_max_phys_footprint instructions cycles "
            "billed_energy serviced_energy interval_max_phys_footprint runnable_time"
        ).split()
    ]


_libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_libproc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
_libproc.proc_pid_rusage.restype = ctypes.c_int


def _process_memory() -> dict[str, float]:
    usage = _ProcessUsage()
    if _libproc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)) != 0:
        raise OSError(ctypes.get_errno(), "Cannot read macOS process footprint")
    return {
        "process_footprint_gib": usage.phys_footprint / 2**30,
        "peak_process_footprint_gib": usage.lifetime_max_phys_footprint / 2**30,
    }


def _memory_snapshot() -> dict[str, float]:
    return {
        "active_memory_gib": mx.get_active_memory() / 2**30,
        "cache_memory_gib": mx.get_cache_memory() / 2**30,
        "peak_active_memory_gib": mx.get_peak_memory() / 2**30,
        **_process_memory(),
    }


def _guard_memory(limit_gib: float, stop: threading.Event) -> None:
    while not stop.wait(0.1):
        memory = _process_memory()
        if memory["process_footprint_gib"] > limit_gib:
            print(
                json.dumps({"memory_guard_aborted": memory, "limit_gib": limit_gib}),
                flush=True,
            )
            os._exit(2)  # Terminate this benchmark before further GPU allocations.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=Path("models/breezeblue/breeze_tts_2/mlx-int8")
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--requests", type=int)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--instruction", default="自然、清晰地朗读，语气温和。")
    parser.add_argument("--language", choices=LINES, default="zh")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--reference-text", type=Path)
    parser.add_argument(
        "--cache-limit-gib",
        type=float,
        default=4.0,
        help="Limit unused MLX buffers retained by this benchmark process.",
    )
    parser.add_argument(
        "--memory-guard-gib",
        type=float,
        default=48.0,
        help="Abort this process if its physical footprint exceeds this threshold.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Also render every request serially for a measured baseline.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    batch_size = args.batch_size or (128 if args.guidance_scale != 1 else 256)
    count = args.requests or batch_size
    if batch_size <= 0 or count <= 0:
        parser.error("batch-size and requests must be positive")
    if (args.reference_audio is None) != (args.reference_text is None):
        parser.error("reference-audio and reference-text must be supplied together")
    if not math.isfinite(args.cache_limit_gib) or args.cache_limit_gib < 0:
        parser.error("cache-limit-gib must be finite and non-negative")
    if (
        not math.isfinite(args.memory_guard_gib)
        or args.memory_guard_gib <= args.cache_limit_gib
    ):
        parser.error("memory-guard-gib must be finite and exceed cache-limit-gib")
    previous_cache_limit = mx.set_cache_limit(int(args.cache_limit_gib * 2**30))
    stop = threading.Event()
    guard = threading.Thread(
        target=_guard_memory, args=(args.memory_guard_gib, stop), daemon=True
    )
    guard.start()
    try:
        _benchmark(args, batch_size, count, previous_cache_limit)
    finally:
        stop.set()
        guard.join()
        mx.set_cache_limit(previous_cache_limit)


def _benchmark(args, batch_size, count, previous_cache_limit):
    model = mlx_speech.tts.load(str(args.model))
    memory_at_load = _memory_snapshot()
    reference = None
    transcript = None
    if args.reference_audio is not None:
        reference, _ = load_audio(args.reference_audio, sample_rate=24_000, mono=True)
        transcript = args.reference_text.read_text(encoding="utf-8").strip()
    texts = [LINES[args.language][i % len(LINES[args.language])] for i in range(count)]
    controls = dict(
        instruction=args.instruction if args.guidance_scale != 1 else None,
        cfg_scale=args.guidance_scale,
        max_frames=args.max_frames,
        top_k=args.top_k,
        temperature=0.9,
        seed=args.seed,
        ref_audio=reference,
        ref_text=transcript,
    )

    def run(size, measured_texts):
        mx.reset_peak_memory()
        timing = {}
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            output = generate_breeze_batch(
                model._speech,
                model._codec,
                model._tokenizer,
                measured_texts,
                batch_size=size,
                timings=timing,
                **controls,
            )
        elapsed = time.perf_counter() - started
        frames = sum(item.frames for item in output)
        audio_seconds = sum(item.samples.size / item.sample_rate for item in output)
        finite = all(bool(mx.all(mx.isfinite(item.samples)).item()) for item in output)
        non_silent = all(
            item.samples.size and float(mx.max(mx.abs(item.samples)).item()) > 1e-5
            for item in output
        )
        return output, dict(
            batch_size=size,
            requests=len(output),
            elapsed_seconds=elapsed,
            audio_seconds=audio_seconds,
            aggregate_realtime=audio_seconds / elapsed,
            frames_per_second=frames / elapsed,
            peak_memory_gib=mx.get_peak_memory() / 2**30,
            finite=finite,
            non_silent=bool(non_silent),
            eos_requests=sum(item.frames < args.max_frames for item in output),
            capped_requests=sum(item.frames == args.max_frames for item in output),
            frames=[item.frames for item in output],
            warnings=[str(w.message) for w in caught],
            **timing,
            **_memory_snapshot(),
        )

    # Warm the actual target shape, including its quantized matmul kernels.
    saved_frames = controls["max_frames"]
    controls["max_frames"] = 2
    run(batch_size, texts)
    controls["max_frames"] = saved_frames
    memory_after_warmup = _memory_snapshot()
    mx.clear_cache()
    print(
        json.dumps(
            {"warmup_complete": True, "memory_before_measurement": _memory_snapshot()}
        ),
        flush=True,
    )
    output, batched = run(batch_size, texts)
    print(
        json.dumps(
            {
                "batch": {
                    key: value
                    for key, value in batched.items()
                    if key not in {"frames", "warnings"}
                }
            }
        ),
        flush=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i in sorted({0, min(1, count - 1), min(2, count - 1), count - 1}):
        write_wav(
            args.output.with_name(f"{args.output.stem}-{i}.wav"),
            output[i].samples,
            sample_rate=output[i].sample_rate,
        )
    report = dict(
        model=str(args.model.resolve()),
        device=mx.device_info(),
        mlx=mx.__version__,
        language=args.language,
        texts=texts,
        guidance_scale=args.guidance_scale,
        instruction=controls["instruction"],
        temperature=0.9,
        reference_audio=str(args.reference_audio) if args.reference_audio else None,
        max_frames=args.max_frames,
        top_k=args.top_k,
        seed=args.seed,
        cache_limit_gib=args.cache_limit_gib,
        previous_cache_limit_gib=previous_cache_limit / 2**30,
        memory_guard_gib=args.memory_guard_gib,
        memory_at_load=memory_at_load,
        memory_after_warmup=memory_after_warmup,
        batch=batched,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.serial:
        _, serial = run(1, texts)
        report["serial"] = serial
        report["wall_time_speedup"] = (
            serial["elapsed_seconds"] / batched["elapsed_seconds"]
        )
        report["throughput_speedup"] = (
            batched["aggregate_realtime"] / serial["aggregate_realtime"]
        )
        print(
            json.dumps(
                {
                    "serial": {
                        key: value
                        for key, value in serial.items()
                        if key not in {"frames", "warnings"}
                    },
                    "throughput_speedup": report["throughput_speedup"],
                }
            ),
            flush=True,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    mx.clear_cache()
    report["memory_after_clear_cache"] = _memory_snapshot()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
