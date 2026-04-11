"""mlx-speech TTS: unified text-to-speech interface.

Usage::

    import mlx_speech

    model = mlx_speech.tts.load("fish-s2-pro")
    result = model.generate("Hello world!")
    # result.waveform: mx.array, result.sample_rate: int
"""

from .._hub import _TTS_MODELS
from .._hub import get_model_path as _get_model_path
from .._hub import list_models as _list_all
from .._hub import resolve_codec_path as _resolve_codec_path
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
    # Alias-first resolution: if the user passed a known alias, we already
    # know the family (needed to disambiguate moss-sound-effect from moss-ttsd
    # since both have the same model_type).
    hint_family: str | None = None
    if path_or_hf_repo in _TTS_MODELS:
        repo, _desc, hint_family = _TTS_MODELS[path_or_hf_repo]
        path_or_hf_repo = repo

    model_dir = _get_model_path(path_or_hf_repo, revision=revision)
    family = hint_family or _resolve_tts_family(model_dir)

    if family == "fish_s2_pro":
        from ._adapters.fish_s2_pro import FishS2ProAdapter

        return FishS2ProAdapter.from_dir(model_dir)

    if family == "vibevoice":
        from ._adapters.vibevoice import VibeVoiceAdapter

        return VibeVoiceAdapter.from_dir(model_dir)

    if family == "longcat":
        from ._adapters.longcat import LongCatAdapter

        return LongCatAdapter.from_dir(model_dir)

    if family == "step_audio":
        from ._adapters.step_audio import StepAudioAdapter

        return StepAudioAdapter.from_dir(model_dir)

    if family in ("moss_local", "moss_delay", "moss_sound_effect"):
        codec_dir = _resolve_codec_path(codec_path_or_repo, revision=revision)
        if family == "moss_local":
            from ._adapters.moss_local import MossLocalAdapter

            return MossLocalAdapter.from_dir(model_dir, codec_dir=codec_dir)
        if family == "moss_sound_effect":
            from ._adapters.moss_sound_effect import MossSoundEffectAdapter

            return MossSoundEffectAdapter.from_dir(model_dir, codec_dir=codec_dir)

        from ._adapters.moss_delay import MossDelayAdapter

        return MossDelayAdapter.from_dir(model_dir, codec_dir=codec_dir)

    raise ValueError(f"Unsupported TTS family: {family!r}")
