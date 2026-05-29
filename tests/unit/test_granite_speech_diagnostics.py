from __future__ import annotations

import json
from pathlib import Path

from scripts.generate.granite_speech_asr import (
    TranscriptRecord,
    discover_default_audio_inputs,
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
        },
        {
            "input_path": "bad.wav",
            "output_path": "transcripts/bad.txt",
            "non_empty": False,
            "error": "failed",
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
