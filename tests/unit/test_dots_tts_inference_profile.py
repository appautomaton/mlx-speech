from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from scripts.eval import profile_dots_tts_inference as profile


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        self.value += 1.0
        return self.value


class _Stream:
    def __init__(self) -> None:
        self.closed = False

    def __iter__(self):
        yield SimpleNamespace(waveform=mx.ones(4), num_patches=1)
        yield SimpleNamespace(waveform=mx.ones(4), num_patches=1)

    def close(self) -> None:
        self.closed = True


class _Generator:
    sample_rate = 4

    def synthesize(self, *_args, **_kwargs):
        return SimpleNamespace(waveform=mx.ones(8), num_patches=2)

    def synthesize_stream(self, *_args, **_kwargs):
        return _Stream()


def test_measure_request_covers_batch_and_stream(monkeypatch) -> None:
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 123)
    for path in profile.PATHS:
        result = profile.measure_request(
            _Generator(),
            path=path,
            text="hello",
            reference_audio=Path("reference.wav"),
            seed=42,
            max_audio_patches=2,
            eos_threshold=0.8,
            clock=_Clock(),
        )
        assert result["path"] == path
        assert result["waveform_samples"] == 8
        assert result["waveform_duration_seconds"] == 2.0
        assert result["patch_count"] == 2
        assert result["stop_reason"] == "patch_budget"
        assert result["peak_memory_bytes"] == 123


def _report(seconds: tuple[float, float, float, float]) -> dict:
    cases = []
    for (variant, path), total in zip(
        (("mf", "batch"), ("mf", "stream"), ("soar", "batch"), ("soar", "stream")),
        seconds,
        strict=True,
    ):
        cases.append({"variant": variant, "path": path, "total_seconds": total})
    return {"config": {"locked": True}, "cases": cases}


def test_raw_before_after_comparison_requires_matching_faster_cells() -> None:
    comparison = profile.compare_reports(
        _report((10.0, 10.0, 20.0, 20.0)),
        _report((9.0, 9.5, 19.0, 18.0)),
    )
    assert comparison["passed"]
    assert all(cell["faster"] for cell in comparison["cells"])

    regression = profile.compare_reports(
        _report((10.0, 10.0, 20.0, 20.0)),
        _report((9.0, 10.1, 19.0, 18.0)),
    )
    assert not regression["passed"]

    with pytest.raises(ValueError, match="configurations differ"):
        profile.compare_reports(
            _report((10.0, 10.0, 20.0, 20.0)),
            {"config": {"locked": False}, "cases": []},
        )


def test_run_loads_once_per_variant_path_and_compares_raw_file(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mx, "reset_peak_memory", lambda: None)
    monkeypatch.setattr(mx, "get_peak_memory", lambda: 123)
    monkeypatch.setattr(mx, "clear_cache", lambda: None)
    loaded = []

    def loader(path: Path):
        loaded.append(path)
        return _Generator()

    args = Namespace(
        model_root=Path("models/dots_tts"),
        reference_audio=Path("reference.wav"),
        artifact_class="int8",
        text="hello",
        seed=42,
        max_audio_patches=2,
        eos_threshold=0.8,
        variants=("mf", "soar"),
        paths=("batch", "stream"),
        compare=None,
    )
    before = profile.run(args, generator_loader=loader, clock=_Clock())
    assert len(loaded) == 4
    assert len(before["cases"]) == 4

    before_path = tmp_path / "before.json"
    profile._write_json(before_path, before)
    args.compare = before_path
    after = profile.run(args, generator_loader=loader, clock=_Clock())
    assert not after["comparison"]["passed"]


def test_cli_has_no_platform_or_repetition_controls() -> None:
    args = profile.parse_args(["--output", "result.json"])
    assert args.artifact_class == "int8"
    for flag in (
        "--candidate-id",
        "--disposition-ledger",
        "--capture",
        "--backend",
        "--runs",
        "--warmup-runs",
        "--comparison-contract",
    ):
        with pytest.raises(SystemExit):
            profile.parse_args(["--output", "result.json", flag, "value"])
