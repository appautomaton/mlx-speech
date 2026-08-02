#!/usr/bin/env python3
"""Run the fixed multilingual dots.tts base-versus-int8 quality gate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import re
import sys
import time
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.eval.dots_tts_comparison_contract import (
        load_comparison_contract,
        update_comparison_contract,
    )
except ModuleNotFoundError:  # Direct ``python scripts/eval/...`` execution.
    from dots_tts_comparison_contract import (  # type: ignore[no-redef]
        load_comparison_contract,
        update_comparison_contract,
    )


ROOT = Path(__file__).resolve().parents[2]
BUILD_MATRIX = (
    ("soar", "base"),
    ("soar", "int8"),
    ("mf", "base"),
    ("mf", "int8"),
)
WEIGHT_FILES = (
    "core.safetensors",
    "vocoder.safetensors",
    "speaker.safetensors",
    "latent_stats.safetensors",
)
QUALITY_RECORD_FIELDS = (
    "key",
    "variant",
    "artifact_class",
    "reference_id",
    "language",
    "mode",
    "artifact_digest",
    "reference_sha256",
    "reference_text",
    "target_text",
    "sample_rate",
    "waveform_samples",
    "num_patches",
    "asr_errors",
    "asr_tokens",
    "wer",
    "speaker_cosine",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dots.tts quant-gate manifest schema")
    references = manifest.get("references")
    if not isinstance(references, list) or not references:
        raise ValueError("dots.tts quant-gate manifest needs references")
    identifiers = [item.get("id") for item in references]
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("dots.tts quant-gate reference ids must be non-empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("dots.tts quant-gate reference ids must be unique")
    languages = set()
    for item in references:
        for field in (
            "language",
            "asr_language",
            "reference_text",
            "target_text",
        ):
            if not isinstance(item.get(field), str) or not item[field].strip():
                raise ValueError(f"dots.tts reference {item['id']} needs {field}")
        languages.add(item["language"])
    if not {"en", "zh"}.issubset(languages):
        raise ValueError("dots.tts quant gate requires English and Mandarin cases")
    if manifest.get("modes") != ["continuation", "speaker_only"]:
        raise ValueError(
            "dots.tts quant gate requires continuation and speaker_only modes"
        )
    generation = manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("dots.tts quant gate needs generation settings")
    if int(generation.get("max_audio_patches", 0)) <= 0:
        raise ValueError("dots.tts max_audio_patches must be positive")
    gate = manifest.get("quality_gate")
    if not isinstance(gate, dict):
        raise ValueError("dots.tts quant gate needs quality thresholds")
    for field in (
        "max_absolute_wer_regression",
        "max_speaker_cosine_regression",
    ):
        value = gate.get(field)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise ValueError(f"dots.tts quality threshold {field} must be non-negative")


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def error_tokens(text: str, language: str) -> list[str]:
    normalized = normalize_text(text)
    if language == "zh":
        return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+", normalized)
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", normalized)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for index, reference_token in enumerate(reference, start=1):
        current = [index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis, start=1):
            substitution = previous[hypothesis_index - 1] + (
                reference_token != hypothesis_token
            )
            current.append(
                min(
                    previous[hypothesis_index] + 1,
                    current[hypothesis_index - 1] + 1,
                    substitution,
                )
            )
        previous = current
    return previous[-1]


def error_counts(reference: str, hypothesis: str, language: str) -> tuple[int, int]:
    reference_tokens = error_tokens(reference, language)
    hypothesis_tokens = error_tokens(hypothesis, language)
    if not reference_tokens:
        raise ValueError("dots.tts WER reference contains no evaluation tokens")
    return edit_distance(reference_tokens, hypothesis_tokens), len(reference_tokens)


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 0:
        raise ValueError("speaker embedding cosine requires non-zero vectors")
    return float(np.dot(first, second) / denominator)


def _aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    errors = sum(int(record["asr_errors"]) for record in records)
    tokens = sum(int(record["asr_tokens"]) for record in records)
    if tokens <= 0:
        raise ValueError("dots.tts quant gate has no ASR reference tokens")
    speaker = float(
        np.mean([float(record["speaker_cosine"]) for record in records])
    )
    return {
        "asr_errors": errors,
        "asr_tokens": tokens,
        "wer": errors / tokens,
        "speaker_cosine": speaker,
        "peak_memory_bytes": max(int(record["peak_memory_bytes"]) for record in records),
    }


def summarize_gate(
    records: list[dict[str, Any]],
    *,
    max_wer_regression: float,
    max_speaker_regression: float,
) -> dict[str, Any]:
    expected_cases = {
        (record["variant"], record["reference_id"], record["mode"])
        for record in records
        if record["artifact_class"] == "base"
    }
    int8_cases = {
        (record["variant"], record["reference_id"], record["mode"])
        for record in records
        if record["artifact_class"] == "int8"
    }
    if expected_cases != int8_cases:
        raise ValueError(
            "dots.tts base/int8 case sets differ: "
            f"base_only={sorted(expected_cases - int8_cases)}, "
            f"int8_only={sorted(int8_cases - expected_cases)}"
        )
    variants = {}
    for variant in sorted({record["variant"] for record in records}):
        base = _aggregate(
            [
                record
                for record in records
                if record["variant"] == variant
                and record["artifact_class"] == "base"
            ]
        )
        int8 = _aggregate(
            [
                record
                for record in records
                if record["variant"] == variant
                and record["artifact_class"] == "int8"
            ]
        )
        wer_regression = float(int8["wer"]) - float(base["wer"])
        speaker_regression = float(base["speaker_cosine"]) - float(
            int8["speaker_cosine"]
        )
        variants[variant] = {
            "base": base,
            "int8": int8,
            "wer_regression": wer_regression,
            "speaker_cosine_regression": speaker_regression,
            "passed": (
                wer_regression <= max_wer_regression
                and speaker_regression <= max_speaker_regression
            ),
        }
    base_all = _aggregate(
        [record for record in records if record["artifact_class"] == "base"]
    )
    int8_all = _aggregate(
        [record for record in records if record["artifact_class"] == "int8"]
    )
    overall_wer_regression = float(int8_all["wer"]) - float(base_all["wer"])
    overall_speaker_regression = float(base_all["speaker_cosine"]) - float(
        int8_all["speaker_cosine"]
    )
    passed = (
        all(bool(value["passed"]) for value in variants.values())
        and overall_wer_regression <= max_wer_regression
        and overall_speaker_regression <= max_speaker_regression
    )
    return {
        "thresholds": {
            "max_absolute_wer_regression": max_wer_regression,
            "max_speaker_cosine_regression": max_speaker_regression,
        },
        "variants": variants,
        "overall": {
            "base": base_all,
            "int8": int8_all,
            "wer_regression": overall_wer_regression,
            "speaker_cosine_regression": overall_speaker_regression,
        },
        "passed": passed,
    }


def _artifact_inventory(path: Path) -> dict[str, Any]:
    files = {}
    for name in (*WEIGHT_FILES, "mlx_config.json"):
        file = path / name
        if not file.is_file():
            raise FileNotFoundError(f"dots.tts artifact file is missing: {file}")
        files[name] = {"bytes": file.stat().st_size, "sha256": _sha256(file)}
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = load_manifest(path / "mlx_config.json")
    return {
        "path": str(path),
        "bytes": sum(
            file.stat().st_size for file in path.rglob("*") if file.is_file()
        ),
        "digest": digest,
        "source": metadata["source"],
        "artifact_class": metadata["artifact_class"],
        "quantization": metadata["quantization"],
        "files": files,
    }


def _validate_corpus_lock(
    manifest_path: Path,
    corpus_dir: Path,
) -> dict[str, Any]:
    lock_path = corpus_dir / "manifest.lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(
            "dots.tts eval corpus is not materialized; run "
            "scripts/eval/materialize_clone_eval_macos.py"
        )
    lock = load_manifest(lock_path)
    if lock.get("source_manifest", {}).get("sha256") != _sha256(manifest_path):
        raise ValueError("dots.tts eval corpus lock does not match its manifest")
    for item in lock.get("references", []):
        path = corpus_dir / "references" / f"{item['id']}.wav"
        if not path.is_file() or _sha256(path) != item.get("sha256"):
            raise ValueError(f"dots.tts eval reference integrity failed: {path}")
        item["path"] = str(path)
    validate_manifest(lock)
    return lock


def _speaker_embedding(generator, waveform, sample_rate: int) -> np.ndarray:
    import mlx.core as mx

    features, length = generator.speaker_frontend.features(
        np.asarray(waveform, dtype=np.float32),
        sample_rate=sample_rate,
    )
    embedding = generator.components.speaker_encoder(
        mx.array(features[None], dtype=mx.float32),
        lengths=mx.array([length], dtype=mx.int32),
    )
    mx.eval(embedding)
    return np.asarray(embedding[0], dtype=np.float32)


def _record_key(record: dict[str, Any]) -> str:
    return "/".join(
        (
            record["variant"],
            record["artifact_class"],
            record["reference_id"],
            record["mode"],
        )
    )


def resolve_case_keys(
    manifest: dict[str, Any],
    requested: list[str] | tuple[str, ...],
) -> tuple[str, ...] | None:
    if not requested:
        return None
    valid = {
        "/".join((variant, artifact_class, reference["id"], mode))
        for variant, artifact_class in BUILD_MATRIX
        for reference in manifest["references"]
        for mode in manifest["modes"]
    }
    if len(set(requested)) != len(requested):
        raise ValueError("dots.tts selected quality cases must be unique")
    unexpected = sorted(set(requested) - valid)
    if unexpected:
        raise ValueError(f"unknown dots.tts quality cases: {unexpected}")
    return tuple(requested)


def freeze_quality_evidence(report_path: str | Path) -> dict[str, Any]:
    path = Path(report_path)
    report = load_manifest(path)
    if report.get("schema_version") != 1:
        raise ValueError("unsupported dots.tts frozen quality report schema")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("dots.tts frozen quality report has no records")
    compact_records = []
    for record in records:
        missing = [field for field in QUALITY_RECORD_FIELDS if field not in record]
        if missing:
            raise ValueError(
                f"dots.tts frozen quality record is missing fields: {missing}"
            )
        compact_records.append({field: record[field] for field in QUALITY_RECORD_FIELDS})
    artifacts = {}
    for key, artifact in report.get("artifacts", {}).items():
        artifacts[key] = {
            "digest": artifact["digest"],
            "artifact_class": artifact["artifact_class"],
            "source": artifact["source"],
            "files": artifact["files"],
        }
    if not artifacts:
        raise ValueError("dots.tts frozen quality report has no artifact identities")
    return {
        "report_sha256": _sha256(path),
        "manifest": report["manifest"],
        "corpus_lock": report["corpus_lock"],
        "asr": {
            "path": report["asr"]["path"],
            "weights_sha256": report["asr"]["weights_sha256"],
        },
        "artifacts": artifacts,
        "thresholds": report["gate"]["thresholds"],
        "records": sorted(compact_records, key=_record_key),
    }


def _quality_aggregate(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        raise ValueError("dots.tts quality comparison aggregate is empty")
    errors = sum(int(record["asr_errors"]) for record in records)
    tokens = sum(int(record["asr_tokens"]) for record in records)
    if tokens <= 0:
        raise ValueError("dots.tts quality comparison has no ASR tokens")
    return {
        "asr_errors": errors,
        "asr_tokens": tokens,
        "wer": errors / tokens,
        "speaker_cosine": float(
            np.mean([float(record["speaker_cosine"]) for record in records])
        ),
    }


def compare_quality_payload(
    payload: dict[str, Any],
    contract_path: str | Path,
    *,
    selected_case_keys: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    contract = load_comparison_contract(contract_path)
    baseline = contract["quality"]
    if payload["manifest"]["sha256"] != baseline.get("manifest", {}).get("sha256"):
        raise ValueError("dots.tts quality manifest differs from baseline")
    if payload["corpus_lock"]["sha256"] != baseline.get("corpus_lock", {}).get(
        "sha256"
    ):
        raise ValueError("dots.tts quality corpus differs from baseline")
    if payload["asr"]["weights_sha256"] != baseline.get("asr", {}).get(
        "weights_sha256"
    ):
        raise ValueError("dots.tts quality ASR evaluator differs from baseline")
    for key, artifact in payload["artifacts"].items():
        if artifact["digest"] != baseline.get("artifacts", {}).get(key, {}).get(
            "digest"
        ):
            raise ValueError(f"dots.tts quality artifact differs from baseline: {key}")
    current_records = {_record_key(record): record for record in payload["records"]}
    baseline_records = {
        _record_key(record): record for record in baseline.get("records", [])
    }
    expected_keys = (
        set(selected_case_keys)
        if selected_case_keys is not None
        else set(baseline_records)
    )
    if set(current_records) != expected_keys:
        raise ValueError("dots.tts current quality case set differs from requested scope")
    if not expected_keys.issubset(baseline_records):
        raise ValueError("dots.tts quality contract is missing requested cases")
    for key in sorted(expected_keys):
        current = current_records[key]
        frozen = baseline_records[key]
        if current["reference_sha256"] != frozen["reference_sha256"]:
            raise ValueError(f"dots.tts quality reference differs from baseline: {key}")
        if current["target_text"] != frozen["target_text"]:
            raise ValueError(f"dots.tts quality target differs from baseline: {key}")
    thresholds = baseline["thresholds"]
    groups: dict[str, dict[str, Any]] = {}
    grouped_keys: dict[str, list[str]] = {}
    for key in sorted(expected_keys):
        record = current_records[key]
        group = f"{record['variant']}/{record['artifact_class']}"
        grouped_keys.setdefault(group, []).append(key)
    for group, keys in grouped_keys.items():
        current = _quality_aggregate([current_records[key] for key in keys])
        frozen = _quality_aggregate([baseline_records[key] for key in keys])
        wer_regression = float(current["wer"]) - float(frozen["wer"])
        speaker_regression = float(frozen["speaker_cosine"]) - float(
            current["speaker_cosine"]
        )
        groups[group] = {
            "case_keys": keys,
            "baseline": frozen,
            "current": current,
            "wer_regression": wer_regression,
            "speaker_cosine_regression": speaker_regression,
            "passed": wer_regression
            <= float(thresholds["max_absolute_wer_regression"])
            and speaker_regression
            <= float(thresholds["max_speaker_cosine_regression"]),
        }
    return {
        "contract": str(contract_path),
        "scope": "selected_cases" if selected_case_keys is not None else "full_matrix",
        "thresholds": thresholds,
        "groups": groups,
        "passed": all(bool(group["passed"]) for group in groups.values()),
    }


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    payload = load_manifest(path)
    return {record["key"]: record for record in payload.get("records", [])}


def _save_records(
    path: Path,
    records: dict[str, dict[str, Any]],
) -> None:
    _write_json(path, {"schema_version": 1, "records": list(records.values())})


def generate_build(
    *,
    variant: str,
    artifact_class: str,
    artifact: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    records_path: Path,
    records: dict[str, dict[str, Any]],
    force: bool,
    peak_limit_bytes: int,
    selected_case_keys: set[str] | None = None,
) -> None:
    import mlx.core as mx

    from mlx_speech.audio import load_audio, write_wav
    from mlx_speech.generation.dots_tts import DotsTTSGenerator

    generator = DotsTTSGenerator.from_dir(artifact["path"])
    for reference in manifest["references"]:
        reference_waveform, reference_rate = load_audio(reference["path"], mono=True)
        reference_embedding = _speaker_embedding(
            generator,
            reference_waveform,
            reference_rate,
        )
        for mode in manifest["modes"]:
            candidate = {
                "variant": variant,
                "artifact_class": artifact_class,
                "reference_id": reference["id"],
                "language": reference["language"],
                "asr_language": reference["asr_language"],
                "mode": mode,
            }
            key = _record_key(candidate)
            if selected_case_keys is not None and key not in selected_case_keys:
                continue
            output = (
                output_dir
                / "generated"
                / variant
                / artifact_class
                / mode
                / f"{reference['id']}.wav"
            )
            existing = records.get(key)
            if (
                not force
                and existing is not None
                and existing.get("artifact_digest") == artifact["digest"]
                and output.is_file()
                and existing.get("output_sha256") == _sha256(output)
            ):
                print(f"  [resume] {key}", flush=True)
                continue

            mx.reset_peak_memory()
            baseline = int(mx.get_active_memory())
            started = time.perf_counter()
            result = generator.synthesize(
                reference["target_text"],
                reference_audio=reference["path"],
                reference_text=(
                    reference["reference_text"]
                    if mode == "continuation"
                    else None
                ),
                language=reference["language"],
                max_audio_patches=int(
                    manifest["generation"]["max_audio_patches"]
                ),
                seed=int(manifest["generation"]["seed"]),
                eos_threshold=float(manifest["generation"]["eos_threshold"]),
            )
            mx.eval(result.waveform)
            duration = time.perf_counter() - started
            peak = int(mx.get_peak_memory())
            if peak > peak_limit_bytes:
                raise MemoryError(
                    f"dots.tts generation peak {peak} exceeds {peak_limit_bytes}"
                )
            output.parent.mkdir(parents=True, exist_ok=True)
            write_wav(output, result.waveform, sample_rate=result.sample_rate)
            generated_embedding = _speaker_embedding(
                generator,
                result.waveform,
                result.sample_rate,
            )
            record = {
                **candidate,
                "key": key,
                "artifact_digest": artifact["digest"],
                "reference_sha256": reference["sha256"],
                "reference_text": reference["reference_text"],
                "target_text": reference["target_text"],
                "output_path": str(output),
                "output_sha256": _sha256(output),
                "sample_rate": result.sample_rate,
                "waveform_samples": int(result.waveform.size),
                "waveform_seconds": int(result.waveform.size) / result.sample_rate,
                "num_patches": result.num_patches,
                "generation_seconds": duration,
                "baseline_memory_bytes": baseline,
                "peak_memory_bytes": peak,
                "incremental_peak_bytes": max(0, peak - baseline),
                "speaker_cosine": cosine_similarity(
                    reference_embedding,
                    generated_embedding,
                ),
                "asr_text": None,
                "asr_errors": None,
                "asr_tokens": None,
            }
            records[key] = record
            _save_records(records_path, records)
            print(
                f"  [generated] {key}: patches={result.num_patches}, "
                f"audio={record['waveform_seconds']:.2f}s, "
                f"time={duration:.2f}s, peak={peak / 1024**3:.2f}GiB, "
                f"speaker={record['speaker_cosine']:.4f}",
                flush=True,
            )
    del generator
    gc.collect()
    mx.clear_cache()


def transcribe_outputs(
    *,
    asr_model: Path,
    records: dict[str, dict[str, Any]],
    records_path: Path,
    force: bool,
) -> dict[str, Any]:
    import mlx.core as mx

    from mlx_speech import asr

    asr_inventory = {
        "path": str(asr_model),
        "config_sha256": _sha256(asr_model / "config.json"),
        "weights_sha256": _sha256(asr_model / "model.safetensors"),
        "weights_bytes": (asr_model / "model.safetensors").stat().st_size,
    }
    evaluator_digest = hashlib.sha256(
        json.dumps(asr_inventory, sort_keys=True).encode("utf-8")
    ).hexdigest()
    recognizer = asr.load(str(asr_model))
    mx.reset_peak_memory()
    for record in records.values():
        if (
            not force
            and record.get("asr_text") is not None
            and record.get("asr_evaluator_digest") == evaluator_digest
        ):
            continue
        result = recognizer.generate(
            record["output_path"],
            language=record["asr_language"],
            max_new_tokens=128,
        )
        errors, tokens = error_counts(
            record["target_text"],
            result.text,
            record["language"],
        )
        record.update(
            {
                "asr_evaluator_digest": evaluator_digest,
                "asr_text": result.text,
                "asr_detected_language": result.language,
                "asr_errors": errors,
                "asr_tokens": tokens,
                "wer": errors / tokens,
            }
        )
        _save_records(records_path, records)
        print(
            f"  [asr] {record['key']}: wer={record['wer']:.4f}, "
            f"text={result.text!r}",
            flush=True,
        )
    asr_inventory["peak_memory_bytes"] = int(mx.get_peak_memory())
    del recognizer
    gc.collect()
    mx.clear_cache()
    return asr_inventory


def render_report(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    status = "PASS" if payload.get("passed", gate["passed"]) else "FAIL"
    lines = [
        f"# dots.tts Quantization Gate — {payload['date']}",
        "",
        f"**Verdict: {status}**",
        "",
        "Affine int8 (group size 64) is applied only to eligible native Qwen "
        "Linear/Embedding modules. All other component dtypes match `mlx-base`.",
        "",
        "## Aggregate results",
        "",
        "| Variant | Base WER | Int8 WER | Δ WER | Base speaker cosine | Int8 speaker cosine | Δ cosine | Gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant, result in gate["variants"].items():
        lines.append(
            f"| {variant} | {result['base']['wer']:.4f} | "
            f"{result['int8']['wer']:.4f} | {result['wer_regression']:+.4f} | "
            f"{result['base']['speaker_cosine']:.4f} | "
            f"{result['int8']['speaker_cosine']:.4f} | "
            f"{result['speaker_cosine_regression']:+.4f} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    overall = gate["overall"]
    lines.extend(
        (
            f"| overall | {overall['base']['wer']:.4f} | "
            f"{overall['int8']['wer']:.4f} | "
            f"{overall['wer_regression']:+.4f} | "
            f"{overall['base']['speaker_cosine']:.4f} | "
            f"{overall['int8']['speaker_cosine']:.4f} | "
            f"{overall['speaker_cosine_regression']:+.4f} | {status} |",
            "",
            "Thresholds: WER regression ≤ "
            f"{gate['thresholds']['max_absolute_wer_regression']:.4f}; "
            "speaker-cosine regression ≤ "
            f"{gate['thresholds']['max_speaker_cosine_regression']:.4f}.",
            "Mandarin uses Unicode Han characters as error-rate tokens; English "
            "uses normalized words.",
            "",
            "## Fixed corpus",
            "",
            "| Reference | Language | Reference text | Target text |",
            "| --- | --- | --- | --- |",
        )
    )
    corpus_records = {}
    for record in payload["records"]:
        corpus_records.setdefault(record["reference_id"], record)
    for reference_id, record in sorted(corpus_records.items()):
        reference_text = str(record["reference_text"]).replace("|", "\\|")
        target_text = str(record["target_text"]).replace("|", "\\|")
        lines.append(
            f"| {reference_id} | {record['language']} | {reference_text} | "
            f"{target_text} |"
        )
    lines.extend(
        (
            "",
            "## Artifact size and peak memory",
            "",
            "| Artifact | Size GiB | Peak GiB |",
            "| --- | ---: | ---: |",
        )
    )
    for variant, artifact_class in BUILD_MATRIX:
        artifact = payload["artifacts"][f"{variant}/{artifact_class}"]
        aggregate = gate["variants"][variant][artifact_class]
        lines.append(
            f"| {variant}/{artifact_class} | "
            f"{artifact['bytes'] / 1024**3:.3f} | "
            f"{aggregate['peak_memory_bytes'] / 1024**3:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Per-case evidence",
            "",
            "| Artifact | Reference | Mode | Patches | Seconds | WER | Speaker cosine | ASR text |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        )
    )
    for record in payload["records"]:
        asr_text = str(record["asr_text"]).replace("|", "\\|")
        lines.append(
            f"| {record['variant']}/{record['artifact_class']} | "
            f"{record['reference_id']} | {record['mode']} | "
            f"{record['num_patches']} | {record['waveform_seconds']:.2f} | "
            f"{record['wer']:.4f} | {record['speaker_cosine']:.4f} | "
            f"{asr_text} |"
        )
    lines.extend(
        (
            "",
            "## Provenance",
            "",
            f"- Corpus manifest: `{payload['manifest']['path']}` "
            f"(`{payload['manifest']['sha256']}`)",
            f"- Corpus lock: `{payload['corpus_lock']['path']}` "
            f"(`{payload['corpus_lock']['sha256']}`)",
            f"- ASR evaluator: `{payload['asr']['path']}`; weights "
            f"`{payload['asr']['weights_sha256']}`",
        )
    )
    for variant, artifact_class in BUILD_MATRIX:
        artifact = payload["artifacts"][f"{variant}/{artifact_class}"]
        lines.append(
            f"- `{variant}/{artifact_class}` artifact digest: "
            f"`{artifact['digest']}`; upstream revision "
            f"`{artifact['source']['revision']}`"
        )
    lines.extend(
        (
            f"- Host: `{payload['host']}`",
            f"- Command: `{payload['command']}`",
            "- Failed cases: none.",
            "- Generated/reference audio and local weights are gitignored; this "
            "report contains metrics and hashes only.",
            "- These are locally reproduced measurements, not upstream benchmark claims.",
            "",
        )
    )
    return "\n".join(lines)


def render_focused_report(payload: dict[str, Any]) -> str:
    comparison = payload["comparison"]
    status = "PASS" if comparison["passed"] else "FAIL"
    lines = [
        f"# dots.tts Decoder Precision Gate — {payload['date']}",
        "",
        f"**Verdict: {status}**",
        "",
        "This focused subset settles the decoder precision boundary before later "
        "optimization slices. The complete base-versus-int8 matrix remains a final gate.",
        "",
        "| Artifact | Cases | Δ WER | Δ speaker cosine | Gate |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for name, result in sorted(comparison["groups"].items()):
        lines.append(
            f"| {name} | {len(result['case_keys'])} | "
            f"{result['wer_regression']:+.4f} | "
            f"{result['speaker_cosine_regression']:+.4f} | "
            f"{'PASS' if result['passed'] else 'FAIL'} |"
        )
    lines.extend(
        (
            "",
            "## Per-case evidence",
            "",
            "| Case | Patches | WER | Speaker cosine |",
            "| --- | ---: | ---: | ---: |",
        )
    )
    for record in payload["records"]:
        lines.append(
            f"| {record['key']} | {record['num_patches']} | "
            f"{record['wer']:.4f} | {record['speaker_cosine']:.4f} |"
        )
    lines.extend(
        (
            "",
            f"- Comparison contract: `{comparison['contract']}`",
            f"- Command: `{payload['command']}`",
            "- These are local acceptance measurements, not benchmark claims.",
            "",
        )
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("examples/clone_eval/dots_tts_macos_multilingual_v1.json"),
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("outputs/dots_tts/eval_corpus"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("models/dots_tts"),
    )
    parser.add_argument(
        "--asr-model",
        type=Path,
        default=Path("models/qwen3_asr_1_7b/mlx-int8"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dots_tts/quant_gate"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            f"docs/benchmarks/dots-tts-quant-gate-{date.today().isoformat()}.md"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--peak-memory-limit-gib", type=float, default=16.0)
    parser.add_argument("--case", dest="cases", action="append", default=[])
    parser.add_argument("--comparison-contract", type=Path)
    parser.add_argument("--freeze-comparison-report", type=Path)
    args = parser.parse_args()
    if args.peak_memory_limit_gib <= 0:
        parser.error("--peak-memory-limit-gib must be positive")
    if args.freeze_comparison_report is not None and args.comparison_contract is None:
        parser.error("--freeze-comparison-report requires --comparison-contract")
    if args.freeze_comparison_report is not None and args.cases:
        parser.error("quality baseline freeze does not accept --case")
    return args


def main() -> None:
    args = parse_args()
    if args.freeze_comparison_report is not None:
        evidence = freeze_quality_evidence(args.freeze_comparison_report)
        update_comparison_contract(
            args.comparison_contract,
            section="quality",
            evidence=evidence,
        )
        print(
            json.dumps(
                {
                    "comparison_contract": str(args.comparison_contract),
                    "quality_report_sha256": evidence["report_sha256"],
                    "record_count": len(evidence["records"]),
                },
                indent=2,
            )
        )
        return
    manifest = load_manifest(args.manifest)
    validate_manifest(manifest)
    selected_case_keys = resolve_case_keys(manifest, args.cases)
    selected_case_set = (
        None if selected_case_keys is None else set(selected_case_keys)
    )
    corpus_lock = _validate_corpus_lock(args.manifest, args.corpus_dir)
    selected_builds = (
        BUILD_MATRIX
        if selected_case_keys is None
        else tuple(
            item
            for item in BUILD_MATRIX
            if any(
                key.startswith(f"{item[0]}/{item[1]}/")
                for key in selected_case_keys
            )
        )
    )
    artifacts = {
        f"{variant}/{artifact_class}": _artifact_inventory(
            args.model_root / variant / f"mlx-{artifact_class}"
        )
        for variant, artifact_class in selected_builds
    }
    records_path = args.output_dir / "records.json"
    records = _load_records(records_path)
    if selected_case_set is not None:
        records = {
            key: value for key, value in records.items() if key in selected_case_set
        }
    peak_limit_bytes = round(args.peak_memory_limit_gib * 1024**3)

    import mlx.core as mx

    mx.set_memory_limit(peak_limit_bytes)
    mx.set_cache_limit(2 * 1024**3)
    for variant, artifact_class in selected_builds:
        generate_build(
            variant=variant,
            artifact_class=artifact_class,
            artifact=artifacts[f"{variant}/{artifact_class}"],
            manifest=corpus_lock,
            output_dir=args.output_dir,
            records_path=records_path,
            records=records,
            force=args.force,
            peak_limit_bytes=peak_limit_bytes,
            selected_case_keys=selected_case_set,
        )
    asr_inventory = transcribe_outputs(
        asr_model=args.asr_model,
        records=records,
        records_path=records_path,
        force=args.force,
    )
    gate_config = manifest["quality_gate"]
    if selected_case_keys is None:
        gate = summarize_gate(
            list(records.values()),
            max_wer_regression=float(gate_config["max_absolute_wer_regression"]),
            max_speaker_regression=float(
                gate_config["max_speaker_cosine_regression"]
            ),
        )
    else:
        gate = {
            "scope": "selected_cases",
            "thresholds": {
                "max_absolute_wer_regression": float(
                    gate_config["max_absolute_wer_regression"]
                ),
                "max_speaker_cosine_regression": float(
                    gate_config["max_speaker_cosine_regression"]
                ),
            },
            "passed": True,
        }
    payload = {
        "schema_version": 1,
        "date": date.today().isoformat(),
        "manifest": {
            "path": str(args.manifest),
            "sha256": _sha256(args.manifest),
        },
        "corpus_lock": {
            "path": str(args.corpus_dir / "manifest.lock.json"),
            "sha256": _sha256(args.corpus_dir / "manifest.lock.json"),
        },
        "artifacts": artifacts,
        "asr": asr_inventory,
        "records": sorted(records.values(), key=_record_key),
        "gate": gate,
        "host": f"{platform.platform()} / {platform.machine()}",
        "command": " ".join(sys.argv),
    }
    if args.comparison_contract is not None:
        payload["comparison"] = compare_quality_payload(
            payload,
            args.comparison_contract,
            selected_case_keys=selected_case_keys,
        )
    payload["passed"] = bool(gate["passed"]) and bool(
        payload.get("comparison", {"passed": True})["passed"]
    )
    _write_json(args.output_dir / "report.json", payload)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = (
        render_report(payload)
        if selected_case_keys is None
        else render_focused_report(payload)
    )
    args.report.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            payload.get("comparison", gate),
            indent=2,
        )
    )
    print(f"Report: {args.report}")
    if not payload["passed"]:
        raise SystemExit("dots.tts quality gate failed")


if __name__ == "__main__":
    main()
