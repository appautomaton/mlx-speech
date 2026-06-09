from __future__ import annotations

import ast
from pathlib import Path


def _imports_torch(source: str) -> bool:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module == "torch" or (node.module or "").startswith("torch."):
                return True
    return False


def test_runtime_modules_do_not_import_torch() -> None:
    runtime_root = Path(__file__).resolve().parents[1] / "src" / "mlx_speech"
    bad_files: list[str] = []
    for path in runtime_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if _imports_torch(text):
            bad_files.append(str(path))
    assert bad_files == []
