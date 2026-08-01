from __future__ import annotations

import copy
import json
import wave
from pathlib import Path

import numpy as np
import pytest

from scripts.eval import dots_tts_quant_gate as gate
from scripts.eval import materialize_clone_eval_macos as materializer
from scripts.eval.dots_tts_comparison_contract import update_comparison_contract


MANIFEST = Path("examples/clone_eval/dots_tts_macos_multilingual_v1.json")
CONTRACT = """# Evidence

## Canonical comparison data

```json
{
  "schema_version": 1,
  "status": "pending",
  "performance": null,
  "quality": null
}
```
"""


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


def test_selected_quality_cases_are_exact_unique_manifest_keys() -> None:
    manifest = gate.load_manifest(MANIFEST)
    selected = (
        "mf/base/samantha_en_us/continuation",
        "soar/base/tingting_zh_cn/speaker_only",
    )
    assert gate.resolve_case_keys(manifest, selected) == selected
    assert gate.resolve_case_keys(manifest, ()) is None
    with pytest.raises(ValueError, match="unique"):
        gate.resolve_case_keys(manifest, (selected[0], selected[0]))
    with pytest.raises(ValueError, match="unknown"):
        gate.resolve_case_keys(manifest, ("mf/base/missing/continuation",))


def _comparison_record(
    key: str,
    *,
    errors: int = 1,
    speaker: float = 0.8,
) -> dict:
    variant, artifact_class, reference_id, mode = key.split("/")
    return {
        "key": key,
        "variant": variant,
        "artifact_class": artifact_class,
        "reference_id": reference_id,
        "language": "en",
        "mode": mode,
        "artifact_digest": f"digest-{variant}-{artifact_class}",
        "reference_sha256": f"reference-{reference_id}",
        "reference_text": "Reference words.",
        "target_text": "Target words.",
        "sample_rate": 48_000,
        "waveform_samples": 96_000,
        "num_patches": 2,
        "asr_errors": errors,
        "asr_tokens": 10,
        "wer": errors / 10,
        "speaker_cosine": speaker,
    }


def _quality_evidence(records: list[dict]) -> dict:
    artifact_keys = {
        f"{record['variant']}/{record['artifact_class']}" for record in records
    }
    return {
        "report_sha256": "report",
        "manifest": {"sha256": "manifest"},
        "corpus_lock": {"sha256": "corpus"},
        "asr": {"path": "asr", "weights_sha256": "asr"},
        "artifacts": {
            key: {
                "digest": f"digest-{key.replace('/', '-')}",
                "artifact_class": key.split("/")[1],
                "source": {"revision": "upstream"},
                "files": {},
            }
            for key in artifact_keys
        },
        "thresholds": {
            "max_absolute_wer_regression": 0.01,
            "max_speaker_cosine_regression": 0.02,
        },
        "records": records,
    }


def _quality_payload(records: list[dict]) -> dict:
    artifact_keys = {
        f"{record['variant']}/{record['artifact_class']}" for record in records
    }
    return {
        "manifest": {"sha256": "manifest"},
        "corpus_lock": {"sha256": "corpus"},
        "asr": {"weights_sha256": "asr"},
        "artifacts": {
            key: {"digest": f"digest-{key.replace('/', '-')}"}
            for key in artifact_keys
        },
        "records": records,
    }


def test_quality_comparison_supports_focused_scope_and_fails_regression(
    tmp_path: Path,
) -> None:
    key = "mf/base/samantha_en_us/continuation"
    baseline_record = _comparison_record(key)
    contract = tmp_path / "slice.md"
    contract.write_text(CONTRACT, encoding="utf-8")
    update_comparison_contract(
        contract,
        section="performance",
        evidence={"baseline": {}},
    )
    update_comparison_contract(
        contract,
        section="quality",
        evidence=_quality_evidence([baseline_record]),
    )
    comparison = gate.compare_quality_payload(
        _quality_payload([_comparison_record(key, speaker=0.79)]),
        contract,
        selected_case_keys=(key,),
    )
    assert comparison["passed"]
    assert comparison["groups"]["mf/base"]["speaker_cosine_regression"] == pytest.approx(
        0.01
    )

    failed = gate.compare_quality_payload(
        _quality_payload([_comparison_record(key, errors=2, speaker=0.79)]),
        contract,
        selected_case_keys=(key,),
    )
    assert not failed["passed"]


def test_freeze_quality_evidence_is_compact_and_complete(tmp_path: Path) -> None:
    key = "soar/base/tingting_zh_cn/speaker_only"
    record = _comparison_record(key)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest": {"path": "manifest", "sha256": "manifest"},
                "corpus_lock": {"path": "corpus", "sha256": "corpus"},
                "asr": {"path": "asr", "weights_sha256": "asr"},
                "artifacts": {
                    "soar/base": {
                        "digest": "digest-soar-base",
                        "artifact_class": "base",
                        "source": {"revision": "upstream"},
                        "files": {},
                    }
                },
                "gate": {
                    "thresholds": {
                        "max_absolute_wer_regression": 0.01,
                        "max_speaker_cosine_regression": 0.02,
                    }
                },
                "records": [{**record, "output_path": "ignored.wav"}],
            }
        ),
        encoding="utf-8",
    )
    evidence = gate.freeze_quality_evidence(report)
    assert evidence["records"] == [record]
    assert "output_path" not in evidence["records"][0]
    assert len(evidence["report_sha256"]) == 64
