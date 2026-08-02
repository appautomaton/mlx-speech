"""Lightweight runtime diagnostics helpers."""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import asdict, dataclass

import mlx.core as mx


@dataclass(frozen=True)
class MlxMemorySnapshot:
    label: str
    active_bytes: int | None
    cache_bytes: int | None
    peak_bytes: int | None

    def to_dict(self) -> dict[str, int | str | None]:
        return asdict(self)


def _call_int(name: str) -> int | None:
    fn = getattr(mx, name, None)
    if fn is None:
        return None
    return int(fn())


def snapshot_mlx_memory(label: str) -> MlxMemorySnapshot:
    """Capture a point-in-time MLX memory snapshot."""
    return MlxMemorySnapshot(
        label=label,
        active_bytes=_call_int("get_active_memory"),
        cache_bytes=_call_int("get_cache_memory"),
        peak_bytes=_call_int("get_peak_memory"),
    )


def reset_mlx_peak_memory() -> None:
    fn = getattr(mx, "reset_peak_memory", None)
    if fn is not None:
        fn()


def clear_mlx_cache() -> None:
    fn = getattr(mx, "clear_cache", None)
    if fn is not None:
        fn()


def process_peak_physical_footprint_bytes() -> int | None:
    """Return the macOS kernel's lifetime peak footprint for this process."""

    if sys.platform != "darwin":
        return None

    uint64_fields = (
        "ri_user_time",
        "ri_system_time",
        "ri_pkg_idle_wkups",
        "ri_interrupt_wkups",
        "ri_pageins",
        "ri_wired_size",
        "ri_resident_size",
        "ri_phys_footprint",
        "ri_proc_start_abstime",
        "ri_proc_exit_abstime",
        "ri_child_user_time",
        "ri_child_system_time",
        "ri_child_pkg_idle_wkups",
        "ri_child_interrupt_wkups",
        "ri_child_pageins",
        "ri_child_elapsed_abstime",
        "ri_diskio_bytesread",
        "ri_diskio_byteswritten",
        "ri_cpu_time_qos_default",
        "ri_cpu_time_qos_maintenance",
        "ri_cpu_time_qos_background",
        "ri_cpu_time_qos_utility",
        "ri_cpu_time_qos_legacy",
        "ri_cpu_time_qos_user_initiated",
        "ri_cpu_time_qos_user_interactive",
        "ri_billed_system_time",
        "ri_serviced_system_time",
        "ri_logical_writes",
        "ri_lifetime_max_phys_footprint",
        "ri_instructions",
        "ri_cycles",
        "ri_billed_energy",
        "ri_serviced_energy",
        "ri_interval_max_phys_footprint",
        "ri_runnable_time",
    )

    class RusageInfoV4(ctypes.Structure):
        _fields_ = [
            ("ri_uuid", ctypes.c_uint8 * 16),
            *((name, ctypes.c_uint64) for name in uint64_fields),
        ]

    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pid_rusage = libproc.proc_pid_rusage
        proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        proc_pid_rusage.restype = ctypes.c_int
        usage = RusageInfoV4()
        if proc_pid_rusage(os.getpid(), 4, ctypes.byref(usage)) != 0:
            return None
    except (AttributeError, OSError):
        return None
    return int(usage.ri_lifetime_max_phys_footprint)
