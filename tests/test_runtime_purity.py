from __future__ import annotations

import ast
from pathlib import Path


BANNED_RUNTIME_IMPORTS = {
    "torch",
    "torchaudio",
    "transformers",
    "vllm",
    "librosa",
    "qwen_asr",
    "mlx_lm",
    "mlx_audio",
}
BANNED_RUNTIME_STRINGS = {
    "qwen-asr",
}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_runtime_modules_do_not_import_banned_dependency_stacks() -> None:
    runtime_root = Path(__file__).resolve().parents[1] / "src" / "mlx_speech"
    bad_files: list[str] = []
    for path in runtime_root.rglob("*.py"):
        banned = sorted(_import_roots(path) & BANNED_RUNTIME_IMPORTS)
        if banned:
            bad_files.append(f"{path}: {', '.join(banned)}")
    assert bad_files == []


def test_runtime_modules_do_not_reference_upstream_qwen_asr_distribution() -> None:
    runtime_root = Path(__file__).resolve().parents[1] / "src" / "mlx_speech"
    bad_files: list[str] = []
    for path in runtime_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        matches = sorted(value for value in BANNED_RUNTIME_STRINGS if value in text)
        if matches:
            bad_files.append(f"{path}: {', '.join(matches)}")
    assert bad_files == []
