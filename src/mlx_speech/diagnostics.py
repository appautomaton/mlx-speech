"""Lightweight runtime diagnostics helpers."""

from __future__ import annotations

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
