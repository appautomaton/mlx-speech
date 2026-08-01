from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from scripts.eval import profile_dots_tts_inference as profile
from scripts.eval.dots_tts_comparison_contract import (
    load_comparison_contract,
    update_comparison_contract,
)


CONTRACT = """# Evidence

Keep this prose exactly.

The data lives under `## Canonical comparison data`.

## Canonical comparison data

```json
{
  "schema_version": 1,
  "status": "pending",
  "performance": null,
  "quality": null
}
```

## Slice evidence

Pending.
"""


class _Profiler:
    def reset(self) -> None:
        pass

    def result(self, total_seconds: float) -> dict[str, float]:
        return {"residual": total_seconds}


def _patch_memory(monkeypatch, *, peak: int = 100) -> None:
    monkeypatch.setattr(mx, "get_active_memory", lambda: 20)
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: peak)


def _trial(path: str, total: float, *, stage: float = 0.5) -> dict:
    return {
        "path": path,
        "total_seconds": total,
        "first_output_seconds": total / 4,
        "output_seconds": 2.0,
        "rtf": total / 2,
        "peak_memory_bytes": 100,
        "stage_seconds": {"acoustic": stage, "residual": total - stage},
    }


def test_contract_updates_one_section_and_preserves_markdown(tmp_path: Path) -> None:
    path = tmp_path / "slice.md"
    path.write_text(CONTRACT, encoding="utf-8")
    update_comparison_contract(path, section="performance", evidence={"baseline": {}})
    pending = load_comparison_contract(path, require_complete=False)
    assert pending["status"] == "pending"
    assert pending["quality"] is None
    assert "Keep this prose exactly." in path.read_text(encoding="utf-8")

    update_comparison_contract(path, section="quality", evidence={"records": []})
    complete = load_comparison_contract(path)
    assert complete["status"] == "complete"
    assert complete["performance"] == {"baseline": {}}
    assert complete["quality"] == {"records": []}


def test_contract_fails_closed_on_duplicate_or_incomplete_data(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.md"
    duplicate.write_text(CONTRACT + CONTRACT, encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_comparison_contract(duplicate, require_complete=False)

    pending = tmp_path / "pending.md"
    pending.write_text(CONTRACT, encoding="utf-8")
    with pytest.raises(ValueError, match="not complete"):
        load_comparison_contract(pending)


def test_summary_uses_only_cached_path_medians() -> None:
    assert not hasattr(profile, "BACKENDS")
    trials = [_trial("batch", value) for value in (3.0, 1.0, 2.0)]
    summary = profile.summarize_case(
        "mf",
        "batch",
        {"total_seconds": 4.0},
        trials,
    )
    assert summary["medians"]["total_seconds"] == 2.0
    assert summary["medians"]["stage_seconds"]["acoustic"] == 0.5
    assert summary["trials"] == trials


def test_measurement_covers_batch_and_stream_without_backend_switch(
    monkeypatch,
) -> None:
    class Stream:
        def __init__(self):
            self.closed = False

        def __iter__(self):
            yield SimpleNamespace(
                waveform=mx.ones(4, dtype=mx.float32), num_patches=1
            )
            yield SimpleNamespace(
                waveform=mx.ones(4, dtype=mx.float32), num_patches=1
            )

        def close(self) -> None:
            self.closed = True

    class Generator:
        sample_rate = 4

        def synthesize(self, *_args, **_kwargs):
            return SimpleNamespace(
                waveform=mx.ones(8, dtype=mx.float32), num_patches=2
            )

        def synthesize_stream(self, *_args, **_kwargs):
            return Stream()

    _patch_memory(monkeypatch)
    generator = Generator()
    for path in profile.PATHS:
        result = profile.measure_trial(
            generator,
            _Profiler(),
            path=path,
            text="hello",
            reference_audio=Path("reference.wav"),
            seed=42,
            max_audio_patches=2,
            eos_threshold=1.0,
            memory_limit_bytes=1_000,
        )
        assert result["path"] == path
        assert result["patch_count"] == 2
        assert result["waveform_samples"] == 8
        assert result["output_health"] == {
            "finite": True,
            "non_silent": True,
            "peak_absolute": 1.0,
        }


def test_measurement_rejects_wrong_patch_count_and_memory(monkeypatch) -> None:
    class Generator:
        sample_rate = 4

        def synthesize(self, *_args, **_kwargs):
            return SimpleNamespace(
                waveform=mx.ones(4, dtype=mx.float32), num_patches=1
            )

    _patch_memory(monkeypatch)
    with pytest.raises(RuntimeError, match="expected 2"):
        profile.measure_trial(
            Generator(),
            _Profiler(),
            path="batch",
            text="hello",
            reference_audio=Path("reference.wav"),
            seed=42,
            max_audio_patches=2,
            eos_threshold=1.0,
            memory_limit_bytes=1_000,
        )

    _patch_memory(monkeypatch, peak=1_000)
    with pytest.raises(MemoryError, match="must remain below"):
        profile.measure_trial(
            Generator(),
            _Profiler(),
            path="batch",
            text="hello",
            reference_audio=Path("reference.wav"),
            seed=42,
            max_audio_patches=1,
            eos_threshold=1.0,
            memory_limit_bytes=1_000,
        )


def _payload(seconds: float) -> dict:
    cases = []
    for variant in profile.VARIANTS:
        for path in profile.PATHS:
            cases.append(
                {
                    "variant": variant,
                    "path": path,
                    "medians": {"total_seconds": seconds},
                }
            )
    return {
        "host": {"platform": "test", "machine": "arm64", "processor": ""},
        "reference": {"sha256": "reference"},
        "config": {
            "artifact_class": "base",
            "text": "text",
            "seed": 42,
            "max_audio_patches": 128,
            "eos_threshold": 1.0,
            "warmup_runs": 1,
            "runs": 3,
            "variants": list(profile.VARIANTS),
            "paths": list(profile.PATHS),
        },
        "artifacts": {name: {"digest": name} for name in profile.VARIANTS},
        "cases": cases,
    }


def test_performance_comparison_validates_identity_and_batch_gate(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "slice.md"
    contract.write_text(CONTRACT, encoding="utf-8")
    update_comparison_contract(
        contract,
        section="performance",
        evidence={"report_sha256": "raw", "baseline": _payload(10.0)},
    )
    update_comparison_contract(contract, section="quality", evidence={"records": []})
    current = _payload(6.0)
    comparison = profile.compare_performance(
        current,
        contract,
        minimum_batch_improvement=0.35,
    )
    assert comparison["passed"]
    assert comparison["variants"]["mf"]["improvement"] == pytest.approx(0.4)

    mismatched = copy.deepcopy(current)
    mismatched["reference"]["sha256"] = "changed"
    with pytest.raises(ValueError, match="reference differs"):
        profile.compare_performance(
            mismatched,
            contract,
            minimum_batch_improvement=0.35,
        )
