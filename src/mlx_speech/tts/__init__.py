"""mlx-speech TTS: unified text-to-speech interface.

Usage::

    import mlx_speech

    model = mlx_speech.tts.load("fish-s2-pro")
    result = model.generate("Hello world!")
    # result.waveform: mx.array, result.sample_rate: int
"""

from __future__ import annotations

from pathlib import Path

from .._hub import get_model_path, list_models as _list_all, resolve_codec_path
from ._adapter import TTSModel, TTSOutput
from ._registry import _resolve_tts_family

__all__ = ["load", "list_models", "TTSModel", "TTSOutput"]


def list_models() -> dict[str, tuple[str, str]]:
    """List available TTS models.

    Returns:
        Dict mapping alias → (hf_repo_id, description).
    """
    return _list_all("tts")


def load(
    path_or_hf_repo: str,
    *,
    codec_path_or_repo: str | None = None,
    revision: str | None = None,
) -> TTSModel:
    """Load a TTS model by local path, short alias, or HuggingFace repo ID.

    Args:
        path_or_hf_repo: Local directory, alias (e.g. ``"fish-s2-pro"``),
            or HF repo ID (e.g. ``"appautomaton/vibevoice-mlx"``).
        codec_path_or_repo: For MOSS models that need a separate codec.
            Defaults to ``appautomaton/openmoss-audio-tokenizer-mlx``.
        revision: Optional HF revision (branch, tag, or commit hash).

    Returns:
        A :class:`TTSModel` with a ``.generate(text, ...)`` method.
    """
    model_dir = get_model_path(path_or_hf_repo, revision=revision)
    family = _resolve_tts_family(model_dir)

    if family == "fish_s2_pro":
        from ._adapters.fish_s2_pro import FishS2ProAdapter

        return FishS2ProAdapter.from_dir(model_dir)

    if family == "vibevoice":
        from ._adapters.vibevoice import VibeVoiceAdapter

        return VibeVoiceAdapter.from_dir(model_dir)

    if family == "longcat":
        from ._adapters.longcat import LongCatAdapter

        return LongCatAdapter.from_dir(model_dir)

    if family in ("moss_local", "moss_delay"):
        codec_dir = resolve_codec_path(codec_path_or_repo, revision=revision)
        if family == "moss_local":
            from ._adapters.moss_local import MossLocalAdapter

            return MossLocalAdapter.from_dir(model_dir, codec_dir=codec_dir)

        from ._adapters.moss_delay import MossDelayAdapter

        return MossDelayAdapter.from_dir(model_dir, codec_dir=codec_dir)

    raise ValueError(f"Unsupported TTS family: {family!r}")
