from __future__ import annotations

import json
from pathlib import Path

from scripts.generate.granite_speech_asr import (
    TranscriptRecord,
    discover_default_audio_inputs,
    transcribe_paths,
    transcript_path_for,
    write_summary,
)


def test_granite_transcript_path_is_collision_safe_under_output_dir(tmp_path):
    output_dir = tmp_path / "outputs" / "granite_speech_asr"

    first = transcript_path_for("outputs/smoke/generated/sample.wav", output_dir)
    second = transcript_path_for("outputs/dramabox/sample.wav", output_dir)

    assert first.parent == output_dir / "transcripts"
    assert second.parent == output_dir / "transcripts"
    assert first.name.startswith("sample-")
    assert second.name.startswith("sample-")
    assert first != second


def test_granite_write_summary_records_error_and_non_empty(tmp_path):
    records = [
        TranscriptRecord(
            input_path="input.wav",
            output_path="transcripts/input.txt",
            non_empty=True,
            error=None,
        ),
        TranscriptRecord(
            input_path="bad.wav",
            output_path="transcripts/bad.txt",
            non_empty=False,
            error="failed",
        ),
    ]

    summary = write_summary(records, tmp_path)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload == [
        {
            "input_path": "input.wav",
            "output_path": "transcripts/input.txt",
            "non_empty": True,
            "error": None,
            "prompt_tokens": None,
            "generated_tokens": None,
            "wall_time_s": None,
            "memory_snapshots": None,
        },
        {
            "input_path": "bad.wav",
            "output_path": "transcripts/bad.txt",
            "non_empty": False,
            "error": "failed",
            "prompt_tokens": None,
            "generated_tokens": None,
            "wall_time_s": None,
            "memory_snapshots": None,
        },
    ]


def test_granite_discover_default_audio_inputs_uses_curated_roots(tmp_path):
    smoke = tmp_path / "outputs" / "smoke" / "generated"
    dramabox = tmp_path / "outputs" / "dramabox"
    smoke.mkdir(parents=True)
    dramabox.mkdir(parents=True)
    first = smoke / "a.wav"
    second = dramabox / "b.wav"
    ignored = dramabox / "b.txt"
    first.write_bytes(b"")
    second.write_bytes(b"")
    ignored.write_text("nope", encoding="utf-8")

    paths = discover_default_audio_inputs((smoke, dramabox, Path("missing")))

    assert paths == [first, second]


def test_granite_transcribe_paths_records_memory_telemetry(tmp_path):
    class Result:
        text = "hello"
        tokens = [1, 2]
        prompt_tokens = 3

    class Runtime:
        def transcribe(self, audio_path, *, max_new_tokens, prompt, language):
            assert audio_path == tmp_path / "input.wav"
            assert max_new_tokens == 4
            assert prompt is None
            assert language == "en"
            return Result()

    audio = tmp_path / "input.wav"
    audio.write_bytes(b"")

    records = transcribe_paths(
        Runtime(),
        [audio],
        output_dir=tmp_path / "diag",
        max_new_tokens=4,
        prompt=None,
        language="en",
        memory_telemetry=True,
    )

    assert records[0].prompt_tokens == 3
    assert records[0].generated_tokens == 2
    assert records[0].wall_time_s is not None
    assert records[0].memory_snapshots is not None
    assert [snap["label"] for snap in records[0].memory_snapshots] == [
        "before_transcribe",
        "after_transcribe",
        "after_clear_cache",
    ]
