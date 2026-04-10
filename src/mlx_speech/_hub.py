"""Model path resolution with HuggingFace Hub fallback."""

from __future__ import annotations

from pathlib import Path


_TTS_MODELS: dict[str, tuple[str, str]] = {
    "fish-s2-pro": ("appautomaton/fishaudio-s2-pro-8bit-mlx", "Fish S2 Pro — dual-AR TTS, voice cloning, emotion tags"),
    "vibevoice": ("appautomaton/vibevoice-mlx", "VibeVoice Large — hybrid LLM+diffusion TTS, voice cloning"),
    "longcat": ("appautomaton/longcat-audiodit-3.5b-8bit-mlx", "LongCat AudioDiT — flow-matching diffusion TTS"),
    "moss-local": ("appautomaton/openmoss-tts-local-mlx", "OpenMOSS TTS Local — local-attention multi-VQ TTS"),
    "moss-ttsd": ("appautomaton/openmoss-ttsd-mlx", "OpenMOSS TTS Delay — delay-pattern dialogue TTS"),
}

_ASR_MODELS: dict[str, tuple[str, str]] = {
    "cohere-asr": ("appautomaton/cohere-asr-mlx", "Cohere Transcribe — multilingual ASR"),
}

_ALIASES: dict[str, str] = {
    alias: repo for alias, (repo, _) in {**_TTS_MODELS, **_ASR_MODELS}.items()
}
_ALIASES["moss-tts-local"] = _TTS_MODELS["moss-local"][0]


def list_models(category: str | None = None) -> dict[str, tuple[str, str]]:
    """List available model aliases.

    Args:
        category: ``"tts"``, ``"asr"``, or ``None`` for all.

    Returns:
        Dict mapping alias → (hf_repo_id, description).
    """
    if category == "tts":
        return dict(_TTS_MODELS)
    if category == "asr":
        return dict(_ASR_MODELS)
    return {**_TTS_MODELS, **_ASR_MODELS}

MOSS_CODEC_REPO = "appautomaton/openmoss-audio-tokenizer-mlx"

_DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.py",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "*.jsonl",
    "*.yaml",
    "*.jinja",
]


def _is_local_path(path: str) -> bool:
    return path.startswith((".", "/", "~"))


def get_model_path(
    path_or_hf_repo: str,
    *,
    revision: str | None = None,
    force_download: bool = False,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Resolve a model path — local directory or HuggingFace repo ID.

    Resolution order:
      1. Check alias dict for short names (e.g. "fish-s2-pro")
      2. If local path exists → return it
      3. If it looks like a local path but doesn't exist → FileNotFoundError
      4. Otherwise → snapshot_download from HuggingFace Hub
    """
    resolved = _ALIASES.get(path_or_hf_repo, path_or_hf_repo)

    local = Path(resolved).expanduser()
    if local.exists():
        return local

    if _is_local_path(resolved):
        raise FileNotFoundError(
            f"Local model path not found: {resolved}"
        )

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            resolved,
            revision=revision,
            allow_patterns=allow_patterns or _DEFAULT_ALLOW_PATTERNS,
            force_download=force_download,
        )
    )


def resolve_codec_path(
    codec_path_or_repo: str | None = None,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Resolve the MOSS audio tokenizer codec path."""
    if codec_path_or_repo is not None:
        return get_model_path(
            codec_path_or_repo, revision=revision, force_download=force_download
        )
    return get_model_path(
        MOSS_CODEC_REPO, revision=revision, force_download=force_download
    )
