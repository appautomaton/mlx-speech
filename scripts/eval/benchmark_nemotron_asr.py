#!/usr/bin/env python3
"""Benchmark Nemotron cache-aware encoder work, memory, and RTFx."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
import types
from pathlib import Path

import mlx.core as mx

from mlx_speech.audio import load_audio
from mlx_speech.models.nemotron_asr.checkpoint import (
    load_nemotron_checkpoint,
    load_state_dict_strict,
)
from mlx_speech.models.nemotron_asr.model import NemotronASRModel
from mlx_speech.models.nemotron_asr.streaming import StreamingEncoder

DEFAULT_MODEL = Path("models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16")
DEFAULT_AUDIO = Path(
    ".references/mlx-audio/mlx_audio/stt/tests/mega_asr/fixtures/clean.wav"
)
REFERENCE_SOURCE = Path(
    ".references/mlx-audio/mlx_audio/stt/models/nemotron_asr"
)


def _load_reference_modules() -> tuple[types.ModuleType, types.ModuleType]:
    package_name = "_benchmark_mlx_audio_nemotron"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REFERENCE_SOURCE.resolve())]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    loaded = {}
    for name in ("config", "attention", "conformer", "streaming"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name, REFERENCE_SOURCE / f"{name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load mlx-audio reference module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["conformer"], loaded["streaming"]


def _measure(run, duration: float, repeats: int) -> dict:  # type: ignore[no-untyped-def]
    times = []
    peaks = []
    frames = []
    for _ in range(repeats):
        mx.clear_cache()
        baseline = mx.get_active_memory()
        mx.reset_peak_memory()
        started = time.perf_counter()
        output = run()
        mx.eval(*output)
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        peaks.append(max(0, mx.get_peak_memory() - baseline))
        frames.append(sum(chunk.shape[1] for chunk in output))
    elapsed = statistics.median(times)
    return {
        "seconds": elapsed,
        "audio_seconds": duration,
        "rtfx": duration / elapsed,
        "encoder_frames": frames[0],
        "milliseconds_per_encoder_frame": elapsed * 1000.0 / frames[0],
        "incremental_peak_bytes": int(statistics.median(peaks)),
    }


def benchmark(
    model_dir: Path,
    audio_path: Path,
    *,
    mel_lengths: tuple[int, ...],
    repeats: int,
    context: tuple[int, int],
) -> dict:
    model = NemotronASRModel.from_dir(model_dir)
    waveform, _ = load_audio(audio_path, sample_rate=16_000, mono=True)
    base_features, base_lengths = model.preprocessor(waveform)
    base_features = base_features[:, : int(base_lengths[0].item())]

    reference_conformer, reference_streaming = _load_reference_modules()
    reference_encoder = reference_conformer.Conformer(model.config.encoder)
    checkpoint = load_nemotron_checkpoint(model_dir)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint.state_dict.items()
        if key.startswith("encoder.")
    }
    load_state_dict_strict(reference_encoder, encoder_state)

    class ReferenceWrapper:
        encoder = reference_encoder
        default_att_context_size = context

        @staticmethod
        def apply_prompt(hidden, language):  # type: ignore[no-untyped-def]
            return hidden

    results = []
    for mel_length in mel_lengths:
        repeats_needed = (mel_length + base_features.shape[1] - 1) // base_features.shape[1]
        features = mx.concatenate([base_features] * repeats_needed, axis=1)[
            :, :mel_length
        ]
        duration = mel_length * model.config.preprocessor.hop_length / 16_000
        chunk_mel_frames = (context[1] + 1) * model.config.encoder.subsampling_factor
        mel_chunks = [
            features[:, start : start + chunk_mel_frames]
            for start in range(0, features.shape[1], chunk_mel_frames)
        ]

        def run_ours():  # type: ignore[no-untyped-def]
            stream = StreamingEncoder(model.encoder, att_context_size=context)
            output = []
            for chunk in mel_chunks:
                current = stream.feed(chunk)
                mx.eval(*current)
                output.extend(current)
            tail = stream.feed(features[:, :0], final=True)
            mx.eval(*tail)
            output.extend(tail)
            return output

        def run_reference():  # type: ignore[no-untyped-def]
            output = []
            for current in reference_streaming.stream_encode_chunks(
                ReferenceWrapper(),
                mel_chunks,
                "en-US",
                att_context_size=list(context),
            ):
                mx.eval(current)
                output.append(current)
            return output

        # Compile both paths before measured runs.
        mx.eval(*run_ours())
        mx.eval(*run_reference())
        results.append(
            {
                "mel_frames": mel_length,
                "ours": _measure(run_ours, duration, repeats),
                "mlx_audio": _measure(run_reference, duration, repeats),
            }
        )
    return {
        "model": str(model_dir),
        "audio": str(audio_path),
        "context": list(context),
        "synchronization": "each native encoder chunk",
        "repeats": repeats,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--mel-lengths", type=int, nargs="+", default=(128, 256, 512))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--left-context", type=int, default=56)
    parser.add_argument("--right-context", type=int, default=3)
    args = parser.parse_args()
    report = benchmark(
        args.model_dir,
        args.audio,
        mel_lengths=tuple(args.mel_lengths),
        repeats=args.repeats,
        context=(args.left_context, args.right_context),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
