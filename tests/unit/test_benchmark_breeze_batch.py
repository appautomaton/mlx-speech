"""Breeze profiling reports cached buffers and the OS footprint separately."""

import threading

import pytest

from scripts.eval import benchmark_breeze_batch as benchmark


def test_memory_report_does_not_treat_active_peak_as_process_footprint(monkeypatch):
    gib = 2**30
    monkeypatch.setattr(benchmark.mx, "get_active_memory", lambda: gib)
    monkeypatch.setattr(benchmark.mx, "get_cache_memory", lambda: 2 * gib)
    monkeypatch.setattr(benchmark.mx, "get_peak_memory", lambda: 3 * gib)
    monkeypatch.setattr(
        benchmark,
        "_process_memory",
        lambda: {
            "process_footprint_gib": 4.0,
            "peak_process_footprint_gib": 5.0,
        },
    )
    assert benchmark._memory_snapshot() == {
        "active_memory_gib": 1.0,
        "cache_memory_gib": 2.0,
        "peak_active_memory_gib": 3.0,
        "process_footprint_gib": 4.0,
        "peak_process_footprint_gib": 5.0,
    }


def test_memory_guard_stops_the_benchmark_at_the_process_limit(monkeypatch, capsys):
    monkeypatch.setattr(
        benchmark,
        "_process_memory",
        lambda: {
            "process_footprint_gib": 49.0,
            "peak_process_footprint_gib": 49.0,
        },
    )

    def stop_process(code):
        raise SystemExit(code)

    monkeypatch.setattr(benchmark.os, "_exit", stop_process)
    with pytest.raises(SystemExit) as error:
        benchmark._guard_memory(48.0, threading.Event())
    assert error.value.code == 2
    assert "memory_guard_aborted" in capsys.readouterr().out
