from __future__ import annotations

import ast
from pathlib import Path


SCOPED_FILES = (
    Path("src/mlx_speech/generation/dots_tts.py"),
    Path("src/mlx_speech/tts/_adapters/dots_tts.py"),
    *Path("src/mlx_speech/models/dots_tts").glob("*.py"),
)


def test_dots_tts_runtime_has_no_forbidden_dependency_or_reference_imports() -> None:
    forbidden = {"torch", "torchaudio", "transformers", "mlx_lm"}
    for path in SCOPED_FILES:
        source = path.read_text(encoding="utf-8")
        assert ".references" not in source
        tree = ast.parse(source, filename=str(path))
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".", 1)[0])
        assert forbidden.isdisjoint(imports), f"forbidden import in {path}: {imports}"
