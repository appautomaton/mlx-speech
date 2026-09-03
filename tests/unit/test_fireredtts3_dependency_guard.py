from __future__ import annotations

import ast
from pathlib import Path


SCOPED_FILES = (
    Path("src/mlx_speech/generation/fireredtts3.py"),
    Path("src/mlx_speech/tts/_adapters/fireredtts3.py"),
    *Path("src/mlx_speech/models/fireredtts3").glob("*.py"),
)


def test_fireredtts3_runtime_has_no_torch_or_reference_imports() -> None:
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


def test_project_dependencies_remain_free_of_torch_runtime() -> None:
    source = Path("pyproject.toml").read_text(encoding="utf-8")
    dependencies = source.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "torch" not in dependencies
    assert "torchaudio" not in dependencies
