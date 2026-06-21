"""Model path resolution with HuggingFace Hub fallback."""

from __future__ import annotations

from pathlib import Path


# alias -> (hf_repo_id, description, family_hint)
_TTS_MODELS: dict[str, tuple[str, str, str]] = {
    "fish-s2-pro": (
        "appautomaton/fishaudio-s2-pro-8bit-mlx",
        "Fish S2 Pro — dual-AR TTS, voice cloning, emotion tags",
        "fish_s2_pro",
    ),
    "vibevoice": (
        "appautomaton/vibevoice-mlx",
        "VibeVoice Large — hybrid LLM+diffusion TTS, voice cloning",
        "vibevoice",
    ),
    "longcat": (
        "appautomaton/longcat-audiodit-3.5b-8bit-mlx",
        "LongCat AudioDiT — flow-matching diffusion TTS",
        "longcat",
    ),
    "moss-local": (
        "appautomaton/openmoss-tts-local-mlx",
        "OpenMOSS TTS Local — local-attention multi-VQ TTS",
        "moss_local",
    ),
    "moss-ttsd": (
        "appautomaton/openmoss-ttsd-mlx",
        "OpenMOSS TTS Delay — delay-pattern dialogue TTS",
        "moss_delay",
    ),
    "moss-sound-effect": (
        "appautomaton/openmoss-sound-effect-mlx",
        "OpenMOSS Sound Effect — text-to-sound-effect generation",
        "moss_sound_effect",
    ),
    "step-audio": (
        "appautomaton/step-audio-editx-8bit-mlx",
        "Step-Audio-EditX — voice cloning + audio editing (emotion, style, speed)",
        "step_audio",
    ),
    "dramabox": (
        "appautomaton/dramabox-tts-3.3b-bf16-mlx",
        "DramaBox: Resemble flow-matching diffusion TTS, 48 kHz stereo",
        "dramabox",
    ),
}

_ASR_MODELS: dict[str, tuple[str, str, str]] = {
    "cohere-asr": (
        "appautomaton/cohere-asr-mlx",
        "Cohere Transcribe — multilingual ASR",
        "cohere",
    ),
    "qwen3-asr-1.7b": (
        "appautomaton/qwen3-asr-1.7b-bf16-mlx",
        "Qwen3-ASR-1.7B — English, Chinese, and mixed Chinese/English ASR",
        "qwen3",
    ),
}

_ALIASES: dict[str, str] = {
    alias: entry[0] for alias, entry in {**_TTS_MODELS, **_ASR_MODELS}.items()
}
_ALIASES["moss-tts-local"] = _TTS_MODELS["moss-local"][0]


def list_models(category: str | None = None) -> dict[str, tuple[str, str]]:
    """List available model aliases.

    Args:
        category: ``"tts"``, ``"asr"``, or ``None`` for all.

    Returns:
        Dict mapping alias → (hf_repo_id, description).
    """

    def _strip(models: dict[str, tuple[str, str, str]]) -> dict[str, tuple[str, str]]:
        return {alias: (repo, desc) for alias, (repo, desc, _) in models.items()}

    if category == "tts":
        return _strip(_TTS_MODELS)
    if category == "asr":
        return _strip(_ASR_MODELS)
    return {**_strip(_TTS_MODELS), **_strip(_ASR_MODELS)}

MOSS_CODEC_REPO = "appautomaton/openmoss-audio-tokenizer-mlx"
DRAMABOX_GEMMA_REPO = "appautomaton/gemma-3-12b-it-backbone-4bit-mlx"
# RE-USE / SEMamba speech enhancer DramaBox uses for `denoise_ref=True`.
REUSE_REPO = "appautomaton/re-use-semamba-mlx"

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


# Most published MLX repos host artifacts inside a quantization subdirectory
# (mlx-int8/, mlx-4bit/, mlx-8bit/) rather than at the snapshot root, so the
# snapshot returned by snapshot_download doesn't have config.json directly.
# Listed in priority order — int8 is the default runtime target.
_QUANTIZATION_SUBDIRS: tuple[str, ...] = ("mlx-int8", "mlx-4bit", "mlx-8bit")


def _resolve_snapshot_dir(root: Path) -> Path:
    """Return ``root`` or a recognized quantization subdir containing config.json.

    Preserves the path as-is when ``root/config.json`` exists. Otherwise
    descends into the first known quantization subdirectory that has a
    config.json. Falls back to ``root`` when nothing matches so downstream
    callers can raise their own, clearer error.
    """
    if (root / "config.json").exists():
        return root
    for subdir in _QUANTIZATION_SUBDIRS:
        if (root / subdir / "config.json").exists():
            return root / subdir
    return root


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
      2. If local path exists → return it (descending into a quantization
         subdir when config.json is not directly inside)
      3. If it looks like a local path but doesn't exist → FileNotFoundError
      4. Otherwise → snapshot_download from HuggingFace Hub, then descend
         into a quantization subdir if needed
    """
    resolved = _ALIASES.get(path_or_hf_repo, path_or_hf_repo)

    local = Path(resolved).expanduser()
    if local.exists():
        return _resolve_snapshot_dir(local)

    if _is_local_path(resolved):
        raise FileNotFoundError(
            f"Local model path not found: {resolved}"
        )

    from huggingface_hub import snapshot_download

    snapshot = Path(
        snapshot_download(
            resolved,
            revision=revision,
            allow_patterns=allow_patterns or _DEFAULT_ALLOW_PATTERNS,
            force_download=force_download,
        )
    )
    return _resolve_snapshot_dir(snapshot)


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


def resolve_gemma_backbone_path(
    gemma_path_or_repo: str | None = None,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Resolve the Gemma 3 12B text-encoder backbone DramaBox conditions on.

    Defaults to the published ``appautomaton`` backbone repo and auto-downloads
    it, mirroring how MOSS resolves its separate audio codec.
    """
    if gemma_path_or_repo is not None:
        return get_model_path(
            gemma_path_or_repo, revision=revision, force_download=force_download
        )
    return get_model_path(
        DRAMABOX_GEMMA_REPO, revision=revision, force_download=force_download
    )


def resolve_reuse_path(
    reuse_path_or_repo: str | None = None,
    *,
    revision: str | None = None,
    force_download: bool = False,
) -> Path:
    """Resolve the RE-USE / SEMamba enhancer weights DramaBox uses for
    ``denoise_ref=True``.

    Defaults to the published ``appautomaton`` RE-USE repo and auto-downloads
    it, mirroring how the Gemma backbone resolves its separate weights.
    """
    if reuse_path_or_repo is not None:
        return get_model_path(
            reuse_path_or_repo, revision=revision, force_download=force_download
        )
    return get_model_path(
        REUSE_REPO, revision=revision, force_download=force_download
    )
