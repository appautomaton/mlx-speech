#!/usr/bin/env python3
"""Generate dots.tts speech with the complete batch or streaming MLX surface."""

from __future__ import annotations

import argparse
import time
from itertools import chain
from pathlib import Path

import mlx.core as mx

from mlx_speech import tts
from mlx_speech.audio import write_wav, write_wav_chunks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="dots-tts-soar")
    parser.add_argument("--artifact-subdir", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", "-o", default="outputs/dots_tts.wav")
    parser.add_argument("--reference-audio", default=None)
    parser.add_argument("--reference-text", default=None)
    parser.add_argument("--max-audio-patches", type=int, default=500)
    parser.add_argument("--solver-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--speaker-scale", type=float, default=1.5)
    parser.add_argument("--language", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos-threshold", type=float, default=0.8)
    parser.add_argument(
        "--template",
        choices=("tts", "tts_interleave"),
        default="tts",
    )
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--stream-chunk-patches", type=int, default=4)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _generation_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "max_audio_patches": args.max_audio_patches,
        "guidance_scale": args.guidance_scale,
        "speaker_scale": args.speaker_scale,
        "seed": args.seed,
        "eos_threshold": args.eos_threshold,
        "template": args.template,
    }
    if args.reference_audio is not None:
        kwargs["reference_audio"] = args.reference_audio
    if args.reference_text is not None:
        kwargs["reference_text"] = args.reference_text
    if args.solver_steps is not None:
        kwargs["solver_steps"] = args.solver_steps
    if args.language is not None:
        kwargs["language"] = args.language
    return kwargs


def run(args: argparse.Namespace) -> Path:
    load_started = time.perf_counter()
    model = tts.load(
        args.model,
        artifact_subdir=args.artifact_subdir,
    )
    load_seconds = time.perf_counter() - load_started
    kwargs = _generation_kwargs(args)

    mx.reset_peak_memory()
    request_started = time.perf_counter()
    first_chunk_seconds: float | None = None
    chunk_count = 0
    if args.stream:
        if not isinstance(model, tts.StreamingTTSModel):
            raise ValueError(f"TTS model {args.model!r} does not support streaming")
        stream = iter(
            model.generate_stream(
                args.text,
                stream_chunk_patches=args.stream_chunk_patches,
                **kwargs,
            )
        )
        try:
            first = next(stream)
        except StopIteration as error:
            raise RuntimeError("dots.tts streaming produced no waveform chunks") from error
        first_chunk_seconds = time.perf_counter() - request_started
        sample_rate = first.sample_rate
        sample_count = 0

        def waveforms():
            nonlocal chunk_count, sample_count
            for chunk in chain((first,), stream):
                if chunk.sample_rate != sample_rate:
                    raise RuntimeError(
                        "dots.tts stream changed sample rate: "
                        f"{sample_rate} -> {chunk.sample_rate}"
                    )
                chunk_count += 1
                sample_count += int(chunk.waveform.size)
                yield chunk.waveform

        try:
            output_path = write_wav_chunks(
                args.output,
                waveforms(),
                sample_rate=sample_rate,
            )
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
    else:
        result = model.generate(args.text, **kwargs)
        sample_rate = result.sample_rate
        sample_count = int(result.waveform.size)
        output_path = write_wav(
            args.output,
            result.waveform,
            sample_rate=sample_rate,
        )

    request_seconds = time.perf_counter() - request_started
    audio_seconds = sample_count / sample_rate
    metrics = [
        f"output={output_path}",
        f"sample_rate={sample_rate}",
        f"load={load_seconds:.3f}s",
        f"request={request_seconds:.3f}s",
        f"audio={audio_seconds:.3f}s",
        f"rtf={request_seconds / audio_seconds:.3f}",
    ]
    if first_chunk_seconds is not None:
        metrics.extend(
            (
                f"first_chunk={first_chunk_seconds:.3f}s",
                f"chunks={chunk_count}",
            )
        )
    print("dots.tts " + " ".join(metrics))
    return output_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
