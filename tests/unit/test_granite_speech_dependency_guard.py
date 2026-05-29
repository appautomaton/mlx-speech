from __future__ import annotations

import ast
from pathlib import Path


BANNED_RUNTIME_IMPORTS = {
    "torch",
    "torchaudio",
    "mlx_lm",
    "transformers",
    "vllm",
    "mlx_audio",
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


def test_runtime_source_does_not_import_reference_or_torch_stacks():
    offenders: list[str] = []
    for path in sorted(Path("src/mlx_speech").rglob("*.py")):
        banned = sorted(_import_roots(path) & BANNED_RUNTIME_IMPORTS)
        if banned:
            offenders.append(f"{path}: {', '.join(banned)}")

    assert offenders == []
