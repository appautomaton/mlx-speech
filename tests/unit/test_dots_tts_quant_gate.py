from __future__ import annotations

import copy
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.eval import dots_tts_quant_gate as gate
from scripts.eval import materialize_clone_eval_macos as materializer


MANIFEST = Path("examples/clone_eval/dots_tts_macos_multilingual_v1.json")


def _record(
    artifact_class: str,
    *,
    errors: int,
    speaker: float,
    reference_id: str = "english",
    mode: str = "continuation",
) -> dict:
    return {
        "variant": "soar",
        "artifact_class": artifact_class,
        "reference_id": reference_id,
        "mode": mode,
        "asr_errors": errors,
        "asr_tokens": 10,
        "speaker_cosine": speaker,
        "peak_memory_bytes": 100,
    }


def test_manifest_is_fixed_multilingual_and_bounded() -> None:
    manifest = gate.load_manifest(MANIFEST)
    gate.validate_manifest(manifest)
    materializer.validate_manifest(manifest)
    assert {item["language"] for item in manifest["references"]} == {"en", "zh"}
    assert manifest["modes"] == ["continuation", "speaker_only"]
    assert manifest["generation"]["max_audio_patches"] == 128

    invalid = copy.deepcopy(manifest)
    invalid["references"][1]["language"] = "en"
    with pytest.raises(ValueError, match="English and Mandarin"):
        gate.validate_manifest(invalid)


def test_language_aware_error_counts_are_deterministic() -> None:
    assert gate.error_counts("Today weather", "today the weather", "en") == (1, 2)
    assert gate.error_counts("今天很好。", "今天真好", "zh") == (1, 4)
    with pytest.raises(ValueError, match="no evaluation tokens"):
        gate.error_counts("...", "anything", "en")


def test_gate_compares_matching_aggregate_base_and_int8_records() -> None:
    records = [
        _record("base", errors=1, speaker=0.80),
        _record("int8", errors=1, speaker=0.79),
    ]
    summary = gate.summarize_gate(
        records,
        max_wer_regression=0.01,
        max_speaker_regression=0.02,
    )
    assert summary["passed"]
    assert summary["variants"]["soar"]["wer_regression"] == 0.0
    assert summary["variants"]["soar"]["speaker_cosine_regression"] == pytest.approx(
        0.01
    )

    failed = [records[0], _record("int8", errors=2, speaker=0.79)]
    summary = gate.summarize_gate(
        failed,
        max_wer_regression=0.01,
        max_speaker_regression=0.02,
    )
    assert not summary["passed"]

    mismatched = [
        records[0],
        _record("int8", errors=1, speaker=0.79, reference_id="mandarin"),
    ]
    with pytest.raises(ValueError, match="case sets differ"):
        gate.summarize_gate(
            mismatched,
            max_wer_regression=0.01,
            max_speaker_regression=0.02,
        )


def test_materialized_audio_record_requires_mono_24khz_pcm16(tmp_path: Path) -> None:
    path = tmp_path / "reference.wav"
    samples = (np.linspace(-0.2, 0.2, 240, dtype=np.float32) * 32767).astype(
        np.int16
    )
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(samples.tobytes())
    record = materializer.audio_record(path)
    assert record["frames"] == 240
    assert record["duration_seconds"] == pytest.approx(0.01)
    assert len(str(record["sha256"])) == 64


def test_manifest_json_is_stable_and_has_no_generated_paths() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert all("path" not in item and "sha256" not in item for item in payload["references"])


def test_report_records_prompts_artifact_digests_and_revisions() -> None:
    records = []
    for variant, artifact_class in gate.BUILD_MATRIX:
        records.append(
            {
                **_record(artifact_class, errors=0, speaker=0.8),
                "variant": variant,
                "key": f"{variant}/{artifact_class}/english/continuation",
                "language": "en",
                "reference_text": "Reference words.",
                "target_text": "Target words.",
                "num_patches": 2,
                "waveform_seconds": 0.32,
                "wer": 0.0,
                "asr_text": "Target words.",
            }
        )
    summary = gate.summarize_gate(
        records,
        max_wer_regression=0.01,
        max_speaker_regression=0.02,
    )
    artifacts = {
        f"{variant}/{artifact_class}": {
            "bytes": 1024,
            "digest": f"digest-{variant}-{artifact_class}",
            "source": {"revision": f"revision-{variant}"},
        }
        for variant, artifact_class in gate.BUILD_MATRIX
    }
    report = gate.render_report(
        {
            "date": "2026-07-30",
            "gate": summary,
            "artifacts": artifacts,
            "records": records,
            "manifest": {"path": "manifest.json", "sha256": "manifest-sha"},
            "corpus_lock": {"path": "lock.json", "sha256": "lock-sha"},
            "asr": {"path": "asr", "weights_sha256": "asr-sha"},
            "host": "test-host",
            "command": "test-command",
        }
    )
    assert "Reference words." in report and "Target words." in report
    for artifact in artifacts.values():
        assert artifact["digest"] in report
        assert artifact["source"]["revision"] in report
