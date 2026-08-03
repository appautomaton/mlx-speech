"""TTS generation command handler.

The public entry point is :func:`tts_main` which takes pre-parsed argparse
namespace. A standalone :func:`main` is also provided so
``python -m mlx_speech.tts.generate`` still works.
"""

from __future__ import annotations

import argparse
import sys
import time
from itertools import chain
from pathlib import Path


def add_tts_args(parser: argparse.ArgumentParser) -> None:
    """Register TTS CLI arguments on the given parser."""
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available TTS models and exit.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path, alias (e.g. fish-s2-pro), or HF repo ID.",
    )
    parser.add_argument(
        "--artifact-subdir",
        default=None,
        help="Explicit artifact path inside a shared local or HF model repository.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Text to synthesize (required for most models; optional in edit mode).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.wav",
        help="Output WAV path (default: output.wav).",
    )
    parser.add_argument(
        "--reference-audio",
        default=None,
        help="Reference audio path for voice cloning or editing.",
    )
    parser.add_argument(
        "--reference-text",
        default=None,
        help="Transcript of reference audio.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate.",
    )
    parser.add_argument(
        "--codec",
        default=None,
        help="Codec model path or HF repo (for MOSS models).",
    )
    parser.add_argument(
        "--edit-type",
        default=None,
        choices=[
            "emotion",
            "style",
            "speed",
            "denoise",
            "vad",
            "paralinguistic",
        ],
        help="Audio edit type (Step Audio only).",
    )
    parser.add_argument(
        "--edit-info",
        default=None,
        help="Edit qualifier, e.g. 'happy', 'whispering', 'fast'.",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=None,
        help="Target duration in seconds (Moss Sound Effect).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Write waveform chunks incrementally when the model supports streaming.",
    )
    parser.add_argument(
        "--stream-chunk-patches",
        type=int,
        default=4,
        help="dots.tts patches per steady-state streaming vocoder call.",
    )


def _print_models() -> None:
    from . import list_models

    print("Available TTS models:\n")
    for alias, info in list_models(detailed=True).items():
        print(f"  {alias:<20} {info.description}")
        print(f"  {' ' * 20} {info.repo_id}")
        if info.artifact_subdir is not None:
            print(f"  {' ' * 20} artifact: {info.artifact_subdir}")
        print()


def _format_gib(value: int | None) -> str:
    return "unavailable" if value is None else f"{value / 1024**3:.3f}GiB"


def tts_main(args: argparse.Namespace) -> None:
    """Execute the TTS command with a pre-parsed argparse namespace."""
    if args.list_models:
        _print_models()
        return

    if not args.model:
        print("Error: --model is required.", file=sys.stderr)
        sys.exit(1)

    import mlx.core as mx

    from . import StreamingTTSModel, load
    from ..audio import write_wav, write_wav_chunks
    from ..diagnostics import (
        process_peak_physical_footprint_bytes,
        snapshot_mlx_memory,
    )

    load_started = time.perf_counter()
    model = load(
        args.model,
        artifact_subdir=args.artifact_subdir,
        codec_path_or_repo=args.codec,
    )
    load_seconds = time.perf_counter() - load_started

    generate_kwargs: dict = {}
    if args.reference_audio is not None:
        generate_kwargs["reference_audio"] = args.reference_audio
    if args.reference_text is not None:
        generate_kwargs["reference_text"] = args.reference_text
    if args.max_new_tokens is not None:
        generate_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.edit_type is not None:
        generate_kwargs["edit_type"] = args.edit_type
    if args.edit_info is not None:
        generate_kwargs["edit_info"] = args.edit_info
    if args.duration_seconds is not None:
        generate_kwargs["duration_seconds"] = args.duration_seconds

    mx.reset_peak_memory()
    request_started = time.perf_counter()
    first_chunk_seconds: float | None = None
    if args.stream:
        if not isinstance(model, StreamingTTSModel):
            raise ValueError(f"TTS model {args.model!r} does not support waveform streaming")
        stream = iter(
            model.generate_stream(
                args.text,
                stream_chunk_patches=args.stream_chunk_patches,
                **generate_kwargs,
            )
        )
        try:
            first = next(stream)
        except StopIteration as error:
            raise RuntimeError("TTS streaming produced no waveform chunks") from error
        first_chunk_seconds = time.perf_counter() - request_started
        sample_rate = first.sample_rate
        sample_count = 0

        def waveforms():
            nonlocal sample_count
            for chunk in chain((first,), stream):
                if chunk.sample_rate != sample_rate:
                    raise RuntimeError(
                        "TTS stream changed sample rate: "
                        f"{sample_rate} -> {chunk.sample_rate}"
                    )
                sample_count += int(chunk.waveform.size)
                yield chunk.waveform

        try:
            output_path = write_wav_chunks(
                Path(args.output), waveforms(), sample_rate=sample_rate
            )
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                close()
    else:
        result = model.generate(args.text, **generate_kwargs)
        sample_rate = result.sample_rate
        sample_count = int(result.waveform.size)
        output_path = write_wav(
            Path(args.output), result.waveform, sample_rate=sample_rate
        )
    request_seconds = time.perf_counter() - request_started
    audio_seconds = sample_count / int(sample_rate)
    memory = snapshot_mlx_memory("after_generate")
    process_peak = process_peak_physical_footprint_bytes()
    print(f"Wrote {output_path} (sample_rate={sample_rate})")
    runtime = (
        "Runtime "
        f"load={load_seconds:.3f}s "
        f"request={request_seconds:.3f}s "
        f"total={load_seconds + request_seconds:.3f}s "
        f"audio={audio_seconds:.3f}s "
        f"rtf={request_seconds / audio_seconds:.3f} "
        f"process_peak={_format_gib(process_peak)}"
    )
    if first_chunk_seconds is not None:
        runtime += f" first_chunk={first_chunk_seconds:.3f}s"
    print(runtime)
    print(
        "MLX memory "
        f"active={_format_gib(memory.active_bytes)} "
        f"cache={_format_gib(memory.cache_bytes)} "
        f"peak={_format_gib(memory.peak_bytes)}"
    )


def main() -> None:
    """Standalone entry point for ``python -m mlx_speech.tts.generate``."""
    parser = argparse.ArgumentParser(
        description="Generate speech with mlx-speech TTS models.",
    )
    add_tts_args(parser)
    args = parser.parse_args()
    tts_main(args)


if __name__ == "__main__":
    main()
