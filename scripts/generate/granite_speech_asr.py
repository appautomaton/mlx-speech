#!/usr/bin/env python3
"""Transcribe audio with local Granite Speech ASR checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


DEFAULT_MODEL_DIR = Path("models/ibm/granite_4_0_1b_speech/original")
DEFAULT_OUTPUT_DIR = Path("outputs/granite_speech_asr")
DEFAULT_SAMPLE_ROOTS = (
    Path("outputs/smoke/generated"),
    Path("outputs/dramabox"),
    Path("outputs/clone_eval/manual"),
    Path("outputs/source"),
)


@dataclass(frozen=True)
class TranscriptRecord:
    input_path: str
    output_path: str
    non_empty: bool
    error: str | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    wall_time_s: float | None = None
    memory_snapshots: list[dict[str, int | str | None]] | None = None


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return slug or "audio"


def transcript_path_for(audio_path: str | Path, output_dir: str | Path) -> Path:
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    digest = hashlib.sha1(str(audio_path).encode("utf-8")).hexdigest()[:8]
    filename = f"{_slug(audio_path.stem)}-{digest}.txt"
    return output_dir / "transcripts" / filename


def discover_default_audio_inputs(
    roots: tuple[Path, ...] = DEFAULT_SAMPLE_ROOTS,
) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if root.exists():
            paths.extend(sorted(root.rglob("*.wav")))
    return list(dict.fromkeys(paths))


def write_summary(records: list[TranscriptRecord], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps([asdict(record) for record in records], indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path


def transcribe_paths(
    runtime,
    audio_paths: list[Path],
    *,
    output_dir: Path,
    max_new_tokens: int,
    prompt: str | None,
    language: str,
    memory_telemetry: bool = False,
) -> list[TranscriptRecord]:
    records: list[TranscriptRecord] = []
    transcripts_dir = output_dir / "transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for audio_path in audio_paths:
        output_path = transcript_path_for(audio_path, output_dir)
        started = time.perf_counter()
        snapshots: list[dict[str, int | str | None]] | None = None
        if memory_telemetry:
            from mlx_speech.diagnostics import reset_mlx_peak_memory, snapshot_mlx_memory

            reset_mlx_peak_memory()
            snapshots = [snapshot_mlx_memory("before_transcribe").to_dict()]
        try:
            result = runtime.transcribe(
                audio_path,
                max_new_tokens=max_new_tokens,
                prompt=prompt,
                language=language,
            )
            wall_time_s = time.perf_counter() - started
            if memory_telemetry:
                from mlx_speech.diagnostics import clear_mlx_cache, snapshot_mlx_memory

                snapshots.append(snapshot_mlx_memory("after_transcribe").to_dict())
                clear_mlx_cache()
                snapshots.append(snapshot_mlx_memory("after_clear_cache").to_dict())
            output_path.write_text(result.text + "\n", encoding="utf-8")
            records.append(
                TranscriptRecord(
                    input_path=str(audio_path),
                    output_path=str(output_path),
                    non_empty=bool(result.text.strip()),
                    error=None,
                    prompt_tokens=result.prompt_tokens,
                    generated_tokens=len(result.tokens),
                    wall_time_s=wall_time_s,
                    memory_snapshots=snapshots,
                )
            )
        except Exception as exc:
            wall_time_s = time.perf_counter() - started
            if memory_telemetry:
                from mlx_speech.diagnostics import clear_mlx_cache, snapshot_mlx_memory

                snapshots.append(snapshot_mlx_memory("after_error").to_dict())
                clear_mlx_cache()
                snapshots.append(snapshot_mlx_memory("after_clear_cache").to_dict())
            records.append(
                TranscriptRecord(
                    input_path=str(audio_path),
                    output_path=str(output_path),
                    non_empty=False,
                    error=str(exc),
                    wall_time_s=wall_time_s,
                    memory_snapshots=snapshots,
                )
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Local Granite Speech checkpoint directory.",
    )
    parser.add_argument(
        "--audio",
        nargs="*",
        help="One or more audio files. If omitted, uses curated local output samples.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for transcripts and summary.json.",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--limit", type=int, default=None, help="Limit discovered inputs.")
    parser.add_argument(
        "--memory-telemetry",
        action="store_true",
        help="Record coarse MLX memory snapshots in summary.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    audio_paths = [Path(path) for path in args.audio] if args.audio else discover_default_audio_inputs()
    if args.limit is not None:
        audio_paths = audio_paths[: args.limit]

    if not audio_paths:
        print("Error: no audio inputs found.", file=sys.stderr)
        sys.exit(1)
    missing = [path for path in audio_paths if not path.exists()]
    if missing:
        print(f"Error: audio file not found: {missing[0]}", file=sys.stderr)
        sys.exit(1)

    from mlx_speech.generation.granite_speech_asr import GraniteSpeechAsrModel

    print(f"Loading Granite Speech from {model_dir}", file=sys.stderr)
    runtime = GraniteSpeechAsrModel.from_dir(model_dir)
    records = transcribe_paths(
        runtime,
        audio_paths,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
        language=args.language,
        memory_telemetry=args.memory_telemetry,
    )
    summary_path = write_summary(records, output_dir)
    failures = sum(1 for record in records if record.error or not record.non_empty)
    print(f"Wrote {len(records)} transcript record(s) to {summary_path}", file=sys.stderr)
    if failures:
        print(f"{failures} transcript(s) failed or were empty.", file=sys.stderr)


if __name__ == "__main__":
    main()
