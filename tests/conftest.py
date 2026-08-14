from __future__ import annotations

import os
from pathlib import Path

import pytest


_OPT_IN_TIERS = ("checkpoint", "runtime", "integration")

# Checkpoint and runtime tests skip when local weights are absent. That is the
# right default: it keeps CI green without shipping model files. It also lets a
# slice gate report success having verified nothing, which is how a hard gate
# quietly stops being one. A git worktree makes this easy to hit, since `models/`
# is gitignored and a fresh worktree starts with no weights at all.
#
# Set this to "1" when running a gate. Any skip then fails the session and names
# what was skipped and why.
#
#     MLX_SPEECH_REQUIRE_CHECKPOINTS=1 uv run pytest tests/runtime/test_x.py -q
REQUIRE_CHECKPOINTS_ENV = "MLX_SPEECH_REQUIRE_CHECKPOINTS"


def _requested_tiers(root: Path, args: tuple[str, ...]) -> set[str]:
    enabled: set[str] = set()
    for arg in args:
        if not arg or arg.startswith("-"):
            continue
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        for tier in _OPT_IN_TIERS:
            tier_dir = (root / "tests" / tier).resolve()
            if candidate == tier_dir or tier_dir in candidate.parents:
                enabled.add(tier)
    return enabled


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # type: ignore[no-untyped-def]
    root = Path(str(config.rootpath)).resolve()
    args = tuple(config.invocation_params.args)
    enabled = _requested_tiers(root, args)
    path = Path(str(collection_path)).resolve()
    for tier in _OPT_IN_TIERS:
        tier_dir = (root / "tests" / tier).resolve()
        if path == tier_dir or tier_dir in path.parents:
            return tier not in enabled
    return False


def _skip_reason(report) -> str:  # type: ignore[no-untyped-def]
    """Pull the reason out of a skip report, which pytest stores as a
    ``(path, lineno, message)`` tuple."""
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr) if longrepr else "<no reason given>"


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    """Fail the session on any skip when running a gate.

    See ``REQUIRE_CHECKPOINTS_ENV`` above.
    """
    if os.environ.get(REQUIRE_CHECKPOINTS_ENV) != "1":
        return

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    skipped = reporter.stats.get("skipped", [])
    if not skipped:
        return

    reporter.write_sep(
        "=",
        f"{REQUIRE_CHECKPOINTS_ENV}=1: {len(skipped)} skipped test(s) treated as failures",
        red=True,
    )
    for report in skipped:
        node = getattr(report, "nodeid", "<unknown>")
        reporter.line(f"  SKIPPED {node}: {_skip_reason(report)}", red=True)
    reporter.line(
        "A gate that skips has verified nothing. Stage the weights or drop the flag.",
        red=True,
    )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED
