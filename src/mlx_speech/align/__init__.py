"""mlx-speech alignment: time a known transcript against its audio.

Usage::

    import mlx_speech

    aligner = mlx_speech.align.load("models/qwen3_forced_aligner_0_6b/mlx-int8")
    for item in aligner.generate("speech.wav", "大家好。", language="zh"):
        print(item.text, item.start_s, item.end_s)
"""

from __future__ import annotations

from pathlib import Path

from .._hub import _resolve_snapshot_dir
from ._adapter import AlignItem, AlignModel
from ._registry import _resolve_align_family

__all__ = ["load", "AlignItem", "AlignModel"]


def load(path: str | Path) -> AlignModel:
    """Load a forced-alignment model from a local directory.

    Args:
        path: A converted package such as
            ``models/qwen3_forced_aligner_0_6b/mlx-int8``. A family directory
            resolves to its ``mlx-int8`` package.

    Returns:
        An :class:`AlignModel` whose ``.generate(audio, text, language=...)``
        returns one :class:`AlignItem` per character (Chinese) or word.
    """
    model_dir = Path(path).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Local model path not found: {path}")
    model_dir = _resolve_snapshot_dir(model_dir)
    family = _resolve_align_family(model_dir)

    if family == "qwen3_forced_aligner":
        from ._adapters.qwen3_forced_aligner import Qwen3ForcedAlignerAdapter

        return Qwen3ForcedAlignerAdapter.from_dir(model_dir)

    raise ValueError(f"Unsupported alignment family: {family!r}")
