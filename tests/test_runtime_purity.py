from __future__ import annotations

from pathlib import Path


def test_runtime_modules_do_not_import_torch() -> None:
    runtime_root = Path(__file__).resolve().parents[1] / "src" / "mlx_voice"
    bad_files: list[str] = []
    for path in runtime_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "import torch" in text or "from torch" in text:
            bad_files.append(str(path))
    assert bad_files == []
