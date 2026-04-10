"""mlx-speech ASR: unified automatic speech recognition interface.

Usage::

    import mlx_speech

    model = mlx_speech.asr.load("cohere-asr")
    result = model.generate("audio.wav")
    # result.text: str, result.language: str
"""

from __future__ import annotations

from .._hub import get_model_path, list_models as _list_all
from ._adapter import ASRModel, ASROutput
from ._registry import _resolve_asr_family

__all__ = ["load", "list_models", "ASRModel", "ASROutput"]


def list_models() -> dict[str, tuple[str, str]]:
    """List available ASR models.

    Returns:
        Dict mapping alias → (hf_repo_id, description).
    """
    return _list_all("asr")


def load(
    path_or_hf_repo: str,
    *,
    revision: str | None = None,
) -> ASRModel:
    """Load an ASR model by local path, short alias, or HuggingFace repo ID.

    Args:
        path_or_hf_repo: Local directory, alias (e.g. ``"cohere-asr"``),
            or HF repo ID (e.g. ``"appautomaton/cohere-asr-mlx"``).
        revision: Optional HF revision (branch, tag, or commit hash).

    Returns:
        An :class:`ASRModel` with a ``.generate(audio, ...)`` method.
    """
    model_dir = get_model_path(path_or_hf_repo, revision=revision)
    family = _resolve_asr_family(model_dir)

    if family == "cohere":
        from ._adapters.cohere import CohereASRAdapter

        return CohereASRAdapter.from_dir(model_dir)

    raise ValueError(f"Unsupported ASR family: {family!r}")
