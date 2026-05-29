#!/usr/bin/env python3
"""Run a /tmp-only long-audio Granite Speech ASR benchmark."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.generate.granite_speech_asr import TranscriptRecord, transcribe_paths  # noqa: E402


DEFAULT_MODEL_DIR = Path("models/ibm/granite_4_0_1b_speech/original")
DEFAULT_SOURCE = "three-bears-catamount"


@dataclass(frozen=True)
class BenchmarkSource:
    source_id: str
    title: str
    audio_url: str
    text_url: str
    chapter_title: str
    next_chapter_title: str
    audio_filename: str
    text_filename: str
    script_filename: str
    license: str = "public-domain"


@dataclass(frozen=True)
class AudioChunk:
    index: int
    start_sample: int
    end_sample: int
    sample_rate: int

    @property
    def start_s(self) -> float:
        return self.start_sample / float(self.sample_rate)

    @property
    def end_s(self) -> float:
        return self.end_sample / float(self.sample_rate)

    @property
    def duration_s(self) -> float:
        return (self.end_sample - self.start_sample) / float(self.sample_rate)


SOURCES: dict[str, BenchmarkSource] = {
    DEFAULT_SOURCE: BenchmarkSource(
        source_id=DEFAULT_SOURCE,
        title='The Three Bears of Porcupine Ridge, chapter 6: "Tracked by a Catamount"',
        audio_url=(
            "https://archive.org/download/threebears_porcupineridge_1511_librivox/"
            "threebearsofporcupineridge_06_thompson_64kb.mp3"
        ),
        text_url="https://www.gutenberg.org/cache/epub/49465/pg49465.txt",
        chapter_title="TRACKED BY A CATAMOUNT",
        next_chapter_title="THE CALL OF THE MOOSE",
        audio_filename="tracked_by_a_catamount.mp3",
        text_filename="three_bears_pg49465.txt",
        script_filename="tracked_by_a_catamount_script.txt",
    )
}


def default_output_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="granite-speech-long-audio-", dir="/tmp"))


def ensure_tmp_output_dir(output_dir: Path) -> Path:
    output_dir = output_dir.expanduser().resolve(strict=False)
    tmp_root = Path("/tmp").resolve(strict=True)
    try:
        output_dir.relative_to(tmp_root)
    except ValueError as exc:
        raise ValueError(f"Long-audio benchmark output must be under {tmp_root}; got {output_dir}.") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def download_or_reuse(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "mlx-speech-long-audio-benchmark"},
    )
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(request) as response, tmp_path.open("wb") as out_file:
        shutil.copyfileobj(response, out_file)
    tmp_path.replace(destination)
    return destination


def extract_chapter_text(text: str, *, chapter_title: str, next_chapter_title: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    start = normalized.rfind(chapter_title)
    if start < 0:
        raise ValueError(f"Chapter title not found: {chapter_title}")
    start += len(chapter_title)

    illustration_marker = f"[Illustration: {next_chapter_title}]"
    illustration_end = normalized.find(illustration_marker, start)
    end = normalized.find(next_chapter_title, start)
    if illustration_end >= 0 and (end < 0 or illustration_end < end):
        end = illustration_end
    if end < 0:
        end = normalized.find("*** END", start)
    if end < 0:
        end = len(normalized)

    chapter = normalized[start:end]
    chapter = re.sub(r"\[Illustration:[^\]]+\]", "", chapter)
    return chapter.strip()


def materialize_source(source: BenchmarkSource, output_dir: Path) -> tuple[Path, Path]:
    source_dir = output_dir / "source"
    audio_path = download_or_reuse(source.audio_url, source_dir / source.audio_filename)
    text_path = download_or_reuse(source.text_url, source_dir / source.text_filename)
    script_text = extract_chapter_text(
        text_path.read_text(encoding="utf-8"),
        chapter_title=source.chapter_title,
        next_chapter_title=source.next_chapter_title,
    )
    script_path = source_dir / source.script_filename
    script_path.write_text(script_text + "\n", encoding="utf-8")
    return audio_path, script_path


def plan_chunks(num_samples: int, *, sample_rate: int, chunk_seconds: float) -> list[AudioChunk]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")

    chunk_samples = max(1, int(round(sample_rate * chunk_seconds)))
    chunks: list[AudioChunk] = []
    for index, start in enumerate(range(0, num_samples, chunk_samples)):
        end = min(num_samples, start + chunk_samples)
        chunks.append(
            AudioChunk(
                index=index,
                start_sample=start,
                end_sample=end,
                sample_rate=sample_rate,
            )
        )
    return chunks


def write_chunk_wavs(
    waveform: np.ndarray,
    *,
    chunks: list[AudioChunk],
    output_dir: Path,
    sample_rate: int,
) -> list[Path]:
    from mlx_speech.audio import write_wav
    import mlx.core as mx

    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for chunk in chunks:
        path = chunk_dir / f"chunk-{chunk.index:04d}-{chunk.start_s:08.2f}-{chunk.end_s:08.2f}.wav"
        write_wav(path, mx.array(waveform[chunk.start_sample : chunk.end_sample], dtype=mx.float32), sample_rate=sample_rate)
        paths.append(path)
    return paths


def normalize_words(text: str) -> list[str]:
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", " ").replace("\u2014", " ")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text)


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_word in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _lcs_length(reference: list[str], hypothesis: list[str]) -> int:
    previous = [0] * (len(hypothesis) + 1)
    for ref_word in reference:
        current = [0]
        left = 0
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            value = previous[hyp_index - 1] + 1 if ref_word == hyp_word else max(left, previous[hyp_index])
            current.append(value)
            left = value
        previous = current
    return previous[-1]


def word_metrics(reference_text: str, hypothesis_text: str) -> dict[str, float | int]:
    reference = normalize_words(reference_text)
    hypothesis = normalize_words(hypothesis_text)
    if not reference:
        raise ValueError("reference text has no normalized words")

    edit_distance = _edit_distance(reference, hypothesis)
    lcs = _lcs_length(reference, hypothesis)
    hypothesis_words = len(hypothesis)
    wer = edit_distance / float(len(reference))
    return {
        "reference_words": len(reference),
        "hypothesis_words": hypothesis_words,
        "edit_distance": edit_distance,
        "word_error_rate": wer,
        "word_accuracy": max(0.0, 1.0 - wer),
        "matched_words_lcs": lcs,
        "reference_coverage": lcs / float(len(reference)),
        "hypothesis_precision": lcs / float(hypothesis_words) if hypothesis_words else 0.0,
    }


def _memory_peak_bytes(
    snapshots: list[dict[str, int | str | None]],
    records: list[TranscriptRecord],
) -> int | None:
    peaks: list[int] = []
    for snapshot in snapshots:
        peak = snapshot.get("peak_bytes")
        if isinstance(peak, int):
            peaks.append(peak)
    for record in records:
        for snapshot in record.memory_snapshots or []:
            peak = snapshot.get("peak_bytes")
            if isinstance(peak, int):
                peaks.append(peak)
    return max(peaks) if peaks else None


def _last_snapshot_value(
    snapshots: list[dict[str, int | str | None]],
    key: str,
) -> int | None:
    for snapshot in reversed(snapshots):
        value = snapshot.get(key)
        if isinstance(value, int):
            return value
    return None


def build_summary(
    *,
    source: BenchmarkSource,
    model_dir: Path,
    output_dir: Path,
    audio_path: Path,
    script_path: Path,
    combined_transcript_path: Path,
    chunks: list[AudioChunk],
    records: list[TranscriptRecord],
    memory_snapshots: list[dict[str, int | str | None]],
    reference_text: str,
    hypothesis_text: str,
    total_wall_time_s: float,
    transcribe_wall_time_s: float,
    max_new_tokens: int,
    language: str,
) -> dict[str, object]:
    duration_s = sum(chunk.duration_s for chunk in chunks)
    non_empty_chunks = sum(1 for record in records if record.non_empty)
    prompt_tokens = sum(record.prompt_tokens or 0 for record in records)
    generated_tokens = sum(record.generated_tokens or 0 for record in records)
    rtf = transcribe_wall_time_s / duration_s if duration_s > 0 else None
    rtfx = duration_s / transcribe_wall_time_s if transcribe_wall_time_s > 0 else None
    peak_bytes = _memory_peak_bytes(memory_snapshots, records)

    return {
        "source": {
            "id": source.source_id,
            "title": source.title,
            "license": source.license,
            "audio_url": source.audio_url,
            "text_url": source.text_url,
            "audio_path": str(audio_path),
            "script_path": str(script_path),
        },
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "language": language,
        "duration_s": duration_s,
        "chunk_seconds": max((chunk.duration_s for chunk in chunks), default=0.0),
        "chunk_count": len(chunks),
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "non_empty_chunks": non_empty_chunks,
        "all_chunks_non_empty": non_empty_chunks == len(records),
        "total_wall_time_s": total_wall_time_s,
        "transcribe_wall_time_s": transcribe_wall_time_s,
        "rtf": rtf,
        "rtfx": rtfx,
        "memory": {
            "peak_bytes": peak_bytes,
            "peak_gib": peak_bytes / float(1024**3) if peak_bytes is not None else None,
            "final_active_bytes": _last_snapshot_value(memory_snapshots, "active_bytes"),
            "final_cache_bytes": _last_snapshot_value(memory_snapshots, "cache_bytes"),
        },
        "memory_snapshots": memory_snapshots,
        "combined_transcript_path": str(combined_transcript_path),
        "word_metrics": word_metrics(reference_text, hypothesis_text),
        "chunks": [
            {
                "index": chunk.index,
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
                "duration_s": chunk.duration_s,
                "record": asdict(record),
            }
            for chunk, record in zip(chunks, records, strict=True)
        ],
    }


def write_benchmark_summary(summary: dict[str, object], output_dir: Path) -> Path:
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--source", choices=sorted(SOURCES), default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Benchmark output directory. Defaults to a new directory under /tmp.",
    )
    parser.add_argument("--chunk-seconds", type=float, default=120.0)
    parser.add_argument("--max-new-tokens", type=int, default=350)
    parser.add_argument("--language", default="en")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = SOURCES[args.source]
    output_dir = ensure_tmp_output_dir(Path(args.output_dir) if args.output_dir else default_output_dir())
    model_dir = Path(args.model_dir)

    from mlx_speech.audio import load_audio
    from mlx_speech.diagnostics import clear_mlx_cache, reset_mlx_peak_memory, snapshot_mlx_memory
    from mlx_speech.generation.granite_speech_asr import GraniteSpeechAsrModel

    overall_started = time.perf_counter()
    memory_snapshots: list[dict[str, int | str | None]] = []

    print(f"Using /tmp benchmark directory: {output_dir}", file=sys.stderr)
    audio_path, script_path = materialize_source(source, output_dir)
    reference_text = script_path.read_text(encoding="utf-8")

    reset_mlx_peak_memory()
    memory_snapshots.append(snapshot_mlx_memory("before_model_load").to_dict())
    runtime = GraniteSpeechAsrModel.from_dir(model_dir)
    memory_snapshots.append(snapshot_mlx_memory("after_model_load").to_dict())

    sample_rate = int(runtime.feature_extractor.sample_rate)
    waveform_mx, loaded_sample_rate = load_audio(audio_path, sample_rate=sample_rate, mono=True)
    waveform = np.asarray(waveform_mx, dtype=np.float32)
    del waveform_mx
    memory_snapshots.append(snapshot_mlx_memory("after_audio_load").to_dict())

    chunks = plan_chunks(
        int(waveform.shape[0]),
        sample_rate=int(loaded_sample_rate),
        chunk_seconds=args.chunk_seconds,
    )
    chunk_paths = write_chunk_wavs(
        waveform,
        chunks=chunks,
        output_dir=output_dir,
        sample_rate=int(loaded_sample_rate),
    )
    del waveform
    clear_mlx_cache()
    memory_snapshots.append(snapshot_mlx_memory("after_chunk_materialize").to_dict())

    transcribe_started = time.perf_counter()
    records = transcribe_paths(
        runtime,
        chunk_paths,
        output_dir=output_dir,
        max_new_tokens=args.max_new_tokens,
        prompt=None,
        language=args.language,
        memory_telemetry=True,
    )
    transcribe_wall_time_s = time.perf_counter() - transcribe_started
    memory_snapshots.append(snapshot_mlx_memory("after_all_transcribe").to_dict())
    clear_mlx_cache()
    memory_snapshots.append(snapshot_mlx_memory("after_final_clear_cache").to_dict())

    combined_transcript_path = output_dir / "transcript.txt"
    transcript_text = "\n".join(
        Path(record.output_path).read_text(encoding="utf-8").strip()
        for record in records
        if record.non_empty and Path(record.output_path).exists()
    )
    combined_transcript_path.write_text(transcript_text + "\n", encoding="utf-8")

    summary = build_summary(
        source=source,
        model_dir=model_dir,
        output_dir=output_dir,
        audio_path=audio_path,
        script_path=script_path,
        combined_transcript_path=combined_transcript_path,
        chunks=chunks,
        records=records,
        memory_snapshots=memory_snapshots,
        reference_text=reference_text,
        hypothesis_text=transcript_text,
        total_wall_time_s=time.perf_counter() - overall_started,
        transcribe_wall_time_s=transcribe_wall_time_s,
        max_new_tokens=args.max_new_tokens,
        language=args.language,
    )
    summary_path = write_benchmark_summary(summary, output_dir)

    print(f"Wrote benchmark summary: {summary_path}", file=sys.stderr)
    print(
        "duration={duration:.2f}s chunks={chunks} transcribe_wall={wall:.2f}s "
        "rtfx={rtfx:.2f} peak_gib={peak:.2f} coverage={coverage:.3f} accuracy={accuracy:.3f}".format(
            duration=float(summary["duration_s"]),
            chunks=int(summary["chunk_count"]),
            wall=float(summary["transcribe_wall_time_s"]),
            rtfx=float(summary["rtfx"] or 0.0),
            peak=float((summary["memory"] or {}).get("peak_gib") or 0.0),
            coverage=float(summary["word_metrics"]["reference_coverage"]),
            accuracy=float(summary["word_metrics"]["word_accuracy"]),
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
