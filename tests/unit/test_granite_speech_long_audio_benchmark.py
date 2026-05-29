from __future__ import annotations

from pathlib import Path

from scripts.eval.granite_speech_long_audio import (
    AudioChunk,
    SOURCES,
    build_summary,
    ensure_tmp_output_dir,
    extract_chapter_text,
    normalize_words,
    plan_chunks,
    word_metrics,
)
from scripts.generate.granite_speech_asr import TranscriptRecord


def test_granite_long_audio_plan_chunks_covers_tail():
    chunks = plan_chunks(10, sample_rate=2, chunk_seconds=2.0)

    assert [(chunk.start_sample, chunk.end_sample) for chunk in chunks] == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]
    assert [chunk.duration_s for chunk in chunks] == [2.0, 2.0, 1.0]


def test_granite_long_audio_output_dir_must_be_tmp(tmp_path):
    allowed = ensure_tmp_output_dir(Path("/tmp") / "granite-speech-test-output")
    assert str(allowed).startswith(str(Path("/tmp").resolve()))

    outside = tmp_path / "not-under-tmp"
    try:
        ensure_tmp_output_dir(outside)
    except ValueError as exc:
        assert "must be under" in str(exc)
    else:
        raise AssertionError("expected non-/tmp output dir to fail")


def test_granite_long_audio_extracts_chapter_body_from_gutenberg_text():
    text = """
CONTENTS
VI. TRACKED BY A CATAMOUNT

[Illustration: TRACKED BY A CATAMOUNT]

VI

TRACKED BY A CATAMOUNT

Chapter body line one.
Chapter body line two.

[Illustration: THE CALL OF THE MOOSE]

VII

THE CALL OF THE MOOSE
"""

    chapter = extract_chapter_text(
        text,
        chapter_title="TRACKED BY A CATAMOUNT",
        next_chapter_title="THE CALL OF THE MOOSE",
    )

    assert "Chapter body line one." in chapter
    assert "THE CALL OF THE MOOSE" not in chapter
    assert "Illustration" not in chapter


def test_granite_long_audio_word_metrics_are_text_normalized():
    assert normalize_words("“Gee,” replied Fred--don’t stop!") == [
        "gee",
        "replied",
        "fred",
        "don't",
        "stop",
    ]

    metrics = word_metrics("a b c d", "a x c")

    assert metrics["reference_words"] == 4
    assert metrics["hypothesis_words"] == 3
    assert metrics["edit_distance"] == 2
    assert metrics["word_error_rate"] == 0.5
    assert metrics["word_accuracy"] == 0.5
    assert metrics["matched_words_lcs"] == 2
    assert metrics["reference_coverage"] == 0.5


def test_granite_long_audio_build_summary_records_efficiency_and_memory(tmp_path):
    source = SOURCES["three-bears-catamount"]
    transcript_path = tmp_path / "transcript.txt"
    script_path = tmp_path / "script.txt"
    audio_path = tmp_path / "audio.mp3"
    chunks = [
        AudioChunk(index=0, start_sample=0, end_sample=20, sample_rate=10),
        AudioChunk(index=1, start_sample=20, end_sample=35, sample_rate=10),
    ]
    records = [
        TranscriptRecord(
            input_path="chunk-0.wav",
            output_path="chunk-0.txt",
            non_empty=True,
            prompt_tokens=10,
            generated_tokens=5,
            wall_time_s=1.0,
            memory_snapshots=[
                {"label": "after_transcribe", "active_bytes": 4, "cache_bytes": 5, "peak_bytes": 100}
            ],
        ),
        TranscriptRecord(
            input_path="chunk-1.wav",
            output_path="chunk-1.txt",
            non_empty=True,
            prompt_tokens=8,
            generated_tokens=4,
            wall_time_s=0.75,
            memory_snapshots=[
                {"label": "after_transcribe", "active_bytes": 6, "cache_bytes": 7, "peak_bytes": 90}
            ],
        ),
    ]

    summary = build_summary(
        source=source,
        model_dir=Path("models/granite"),
        output_dir=tmp_path,
        audio_path=audio_path,
        script_path=script_path,
        combined_transcript_path=transcript_path,
        chunks=chunks,
        records=records,
        memory_snapshots=[
            {"label": "after_model_load", "active_bytes": 1, "cache_bytes": 2, "peak_bytes": 50},
            {"label": "after_final_clear_cache", "active_bytes": 3, "cache_bytes": 0, "peak_bytes": 80},
        ],
        reference_text="alpha beta gamma delta",
        hypothesis_text="alpha beta gamma",
        total_wall_time_s=3.0,
        transcribe_wall_time_s=1.75,
        max_new_tokens=32,
        language="en",
    )

    assert summary["duration_s"] == 3.5
    assert summary["chunk_count"] == 2
    assert summary["prompt_tokens"] == 18
    assert summary["generated_tokens"] == 9
    assert summary["rtf"] == 0.5
    assert summary["rtfx"] == 2.0
    assert summary["memory"]["peak_bytes"] == 100
    assert summary["memory"]["final_active_bytes"] == 3
    assert summary["word_metrics"]["reference_coverage"] == 0.75
