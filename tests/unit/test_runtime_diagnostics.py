from __future__ import annotations

import sys

from mlx_speech.diagnostics import process_peak_physical_footprint_bytes
from mlx_speech.tts.generate import _format_gib


def test_process_peak_physical_footprint_uses_the_macos_process_ledger() -> None:
    peak = process_peak_physical_footprint_bytes()
    if sys.platform == "darwin":
        assert peak is not None
        assert peak > 0
    else:
        assert peak is None


def test_memory_log_format_distinguishes_unavailable_values() -> None:
    assert _format_gib(None) == "unavailable"
    assert _format_gib(3 * 1024**3) == "3.000GiB"
