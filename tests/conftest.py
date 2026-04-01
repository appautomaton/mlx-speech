from __future__ import annotations

from pathlib import Path


_OPT_IN_TIERS = ("checkpoint", "runtime", "integration")


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


def _is_explicit_target(path: Path, root: Path, args: tuple[str, ...]) -> bool:
    for arg in args:
        if not arg or arg.startswith("-"):
            continue
        candidate = Path(arg)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        if candidate == path:
            return True
    return False


def pytest_ignore_collect(collection_path: Path, config) -> bool:  # type: ignore[no-untyped-def]
    root = Path(str(config.rootpath)).resolve()
    args = tuple(config.invocation_params.args)
    enabled = _requested_tiers(root, args)
    path = Path(str(collection_path)).resolve()
    for tier in _OPT_IN_TIERS:
        tier_dir = (root / "tests" / tier).resolve()
        if path == tier_dir or tier_dir in path.parents:
            return tier not in enabled
    if path.parent == (root / "tests").resolve() and path.suffix == ".py":
        text = path.read_text(encoding="utf-8")
        if (
            'Path("models/' in text
            or 'MODEL_DIR = "models/' in text
            or "pytest.mark.local_integration" in text
        ):
            return not _is_explicit_target(path, root, args)
    return False
