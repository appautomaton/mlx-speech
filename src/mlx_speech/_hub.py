"""Model path resolution with HuggingFace Hub fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class _ModelAlias:
    repo_id: str
    description: str
    family_hint: str
    artifact_subdir: str | None = None


@dataclass(frozen=True)
class ModelInfo:
    """Public model-catalog entry for one loadable alias."""

    repo_id: str
    description: str
    artifact_subdir: str | None = None


_TTS_MODELS: dict[str, _ModelAlias] = {
    "fish-s2-pro": _ModelAlias(
        "appautomaton/fishaudio-s2-pro-8bit-mlx",
        "Fish S2 Pro — dual-AR TTS, voice cloning, emotion tags",
        "fish_s2_pro",
    ),
    "vibevoice": _ModelAlias(
        "appautomaton/vibevoice-mlx",
        "VibeVoice Large — hybrid LLM+diffusion TTS, voice cloning",
        "vibevoice",
    ),
    "longcat": _ModelAlias(
        "appautomaton/longcat-audiodit-3.5b-8bit-mlx",
        "LongCat AudioDiT — flow-matching diffusion TTS",
        "longcat",
    ),
    "moss-local": _ModelAlias(
        "appautomaton/openmoss-tts-local-mlx",
        "OpenMOSS TTS Local — local-attention multi-VQ TTS",
        "moss_local",
    ),
    "moss-ttsd": _ModelAlias(
        "appautomaton/openmoss-ttsd-mlx",
        "OpenMOSS TTS Delay — delay-pattern dialogue TTS",
        "moss_delay",
    ),
    "moss-sound-effect": _ModelAlias(
        "appautomaton/openmoss-sound-effect-mlx",
        "OpenMOSS Sound Effect — text-to-sound-effect generation",
        "moss_sound_effect",
    ),
    "step-audio": _ModelAlias(
        "appautomaton/step-audio-editx-8bit-mlx",
        "Step-Audio-EditX — voice cloning + audio editing (emotion, style, speed)",
        "step_audio",
    ),
    "dramabox": _ModelAlias(
        "appautomaton/dramabox-tts-3.3b-bf16-mlx",
        "DramaBox: Resemble flow-matching diffusion TTS, 48 kHz stereo",
        "dramabox",
    ),
    "dots-tts-soar": _ModelAlias(
        "appautomaton/dots-tts-mlx",
        "dots.tts SOAR (mlx-int8) — selective-int8 TTS and voice cloning",
        "dots_tts",
        "soar/mlx-int8",
    ),
    "dots-tts-soar-base": _ModelAlias(
        "appautomaton/dots-tts-mlx",
        "dots.tts SOAR base — source-faithful mixed-precision TTS",
        "dots_tts",
        "soar/mlx-base",
    ),
    "dots-tts-soar-int8": _ModelAlias(
        "appautomaton/dots-tts-mlx",
        "dots.tts SOAR int8 — Qwen-selective affine int8 TTS",
        "dots_tts",
        "soar/mlx-int8",
    ),
    "dots-tts-mf": _ModelAlias(
        "appautomaton/dots-tts-mlx",
        "dots.tts MeanFlow (mlx-int8) — selective-int8 TTS and voice cloning",
        "dots_tts",
        "mf/mlx-int8",
    ),
    "dots-tts-mf-base": _ModelAlias(
        "appautomaton/dots-tts-mlx",
        "dots.tts MeanFlow base — source-faithful mixed-precision TTS",
        "dots_tts",
        "mf/mlx-base",
    ),
    "dots-tts-mf-int8": _ModelAlias(
        "appautomaton/dots-tts-mlx",
        "dots.tts MeanFlow int8 — Qwen-selective affine int8 TTS",
        "dots_tts",
        "mf/mlx-int8",
    ),
}

_ASR_MODELS: dict[str, _ModelAlias] = {
    "cohere-asr": _ModelAlias(
        "appautomaton/cohere-asr-mlx",
        "Cohere Transcribe — multilingual ASR",
        "cohere",
    ),
    # Default points at the published int8 build; bf16 stays available via the
    # explicit ``qwen3-asr-1.7b-bf16`` alias.
    "qwen3-asr-1.7b": _ModelAlias(
        "appautomaton/qwen3-asr-1.7b-int8-mlx",
        "Qwen3-ASR-1.7B (int8) — English, Chinese, and mixed Chinese/English ASR",
        "qwen3",
    ),
    "qwen3-asr-1.7b-bf16": _ModelAlias(
        "appautomaton/qwen3-asr-1.7b-bf16-mlx",
        "Qwen3-ASR-1.7B (bf16) — English, Chinese, and mixed Chinese/English ASR",
        "qwen3",
    ),
    "qwen3-asr-1.7b-int8": _ModelAlias(
        "appautomaton/qwen3-asr-1.7b-int8-mlx",
        "Qwen3-ASR-1.7B (int8, affine) — English, Chinese, and mixed Chinese/English ASR",
        "qwen3",
    ),
    # Int8 is the only published Nemotron build. A temporary long-form English
    # and Mandarin comparison found negligible accuracy differences from bf16.
    "nemotron-asr-streaming": _ModelAlias(
        "appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx",
        "Nemotron 3.5 ASR Streaming (int8) — cache-aware multilingual ASR",
        "nemotron",
    ),
    "nemotron-asr-streaming-int8": _ModelAlias(
        "appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx",
        "Nemotron 3.5 ASR Streaming (int8, affine) — cache-aware multilingual ASR",
        "nemotron",
    ),
    # Granite publishes one selective-int8 build. The acoustic encoder and
    # QFormer stay BF16; only the Granite causal LM uses affine int8 weights.
    "granite-speech-4.0-1b": _ModelAlias(
        "appautomaton/granite-4.0-1b-speech-int8-mlx",
        "Granite Speech 4.0 1B (int8) — six-language offline ASR",
        "granite",
    ),
    "granite-speech-4.0-1b-int8": _ModelAlias(
        "appautomaton/granite-4.0-1b-speech-int8-mlx",
        "Granite Speech 4.0 1B (selective affine int8) — six-language offline ASR",
        "granite",
    ),
    # mxfp8 is a supported conversion mode (scripts/convert/qwen3_asr.py --quant
    # mxfp8) but is not published as a downloadable repo — at 8-bit it offers no
    # advantage over int8. Load a locally-built mxfp8 package by path; the
    # mlx-mxfp8 subdir below lets the path resolver auto-descend into it.
}

_ALIASES: dict[str, _ModelAlias] = {
    **_TTS_MODELS,
    **_ASR_MODELS,
}
_ALIASES["moss-tts-local"] = _TTS_MODELS["moss-local"]


def list_models(
    category: str | None = None,
    *,
    detailed: bool = False,
) -> dict[str, tuple[str, str]] | dict[str, ModelInfo]:
    """List available model aliases.

    Args:
        category: ``"tts"``, ``"asr"``, or ``None`` for all.

    Returns:
        Dict mapping alias → ``(hf_repo_id, description)``. With
        ``detailed=True``, values are :class:`ModelInfo` objects that preserve
        a shared repository's artifact subdirectory.
    """

    def _strip(models: dict[str, _ModelAlias]) -> dict[str, tuple[str, str]]:
        return {
            alias: (entry.repo_id, entry.description)
            for alias, entry in models.items()
        }

    def _details(models: dict[str, _ModelAlias]) -> dict[str, ModelInfo]:
        return {
            alias: ModelInfo(
                repo_id=entry.repo_id,
                description=entry.description,
                artifact_subdir=entry.artifact_subdir,
            )
            for alias, entry in models.items()
        }

    render = _details if detailed else _strip

    if category == "tts":
        return render(_TTS_MODELS)
    if category == "asr":
        return render(_ASR_MODELS)
    return {**render(_TTS_MODELS), **render(_ASR_MODELS)}

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
_QUANTIZATION_SUBDIRS: tuple[str, ...] = ("mlx-int8", "mlx-mxfp8", "mlx-4bit", "mlx-8bit")


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


def _normalize_artifact_subdir(artifact_subdir: str) -> str:
    """Validate and normalize a repository-relative artifact selector."""

    if not artifact_subdir or artifact_subdir != artifact_subdir.strip():
        raise ValueError("artifact_subdir must be a non-empty relative POSIX path")
    if "\\" in artifact_subdir or artifact_subdir.startswith("/"):
        raise ValueError("artifact_subdir must be a relative POSIX path")
    parts = artifact_subdir.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(
            "artifact_subdir must not contain empty, '.' or '..' path segments"
        )
    return "/".join(parts)


def _shared_artifact_aliases(repo_id: str) -> dict[str, str]:
    return {
        alias: entry.artifact_subdir
        for alias, entry in _ALIASES.items()
        if entry.repo_id == repo_id and entry.artifact_subdir is not None
    }


def get_model_path(
    path_or_hf_repo: str,
    *,
    artifact_subdir: str | None = None,
    revision: str | None = None,
    force_download: bool = False,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Resolve a model path — local directory or HuggingFace repo ID.

    Resolution order:
      1. Check alias dict for short names (e.g. "fish-s2-pro")
      2. Apply an alias-owned or explicit artifact subdirectory selector
      3. If local path exists → return it (descending into a quantization
         subdir when config.json is not directly inside)
      4. If it looks like a local path but doesn't exist → FileNotFoundError
      5. Otherwise → snapshot_download from HuggingFace Hub, then descend
         into a quantization subdir if needed
    """
    alias = _ALIASES.get(path_or_hf_repo)
    resolved = alias.repo_id if alias is not None else path_or_hf_repo
    explicit_subdir = (
        _normalize_artifact_subdir(artifact_subdir)
        if artifact_subdir is not None
        else None
    )
    alias_subdir = alias.artifact_subdir if alias is not None else None
    if (
        explicit_subdir is not None
        and alias_subdir is not None
        and explicit_subdir != alias_subdir
    ):
        raise ValueError(
            f"Alias {path_or_hf_repo!r} selects {alias_subdir!r}, which conflicts "
            f"with artifact_subdir={explicit_subdir!r}"
        )
    selected_subdir = explicit_subdir or alias_subdir

    local = Path(resolved).expanduser()
    if local.exists():
        if selected_subdir is not None:
            selected = local.joinpath(*selected_subdir.split("/"))
            if not selected.is_dir():
                raise FileNotFoundError(
                    f"Artifact subdirectory not found: {selected_subdir!r} under {local}"
                )
            return _resolve_snapshot_dir(selected)
        return _resolve_snapshot_dir(local)

    if _is_local_path(resolved):
        raise FileNotFoundError(
            f"Local model path not found: {resolved}"
        )

    if selected_subdir is None:
        shared_aliases = _shared_artifact_aliases(resolved)
        if len(set(shared_aliases.values())) > 1:
            choices = ", ".join(
                f"{name} ({subdir})" for name, subdir in sorted(shared_aliases.items())
            )
            raise ValueError(
                f"Hugging Face repo {resolved!r} contains multiple runtime artifacts. "
                f"Load an alias or pass artifact_subdir. Available: {choices}"
            )

    from huggingface_hub import snapshot_download

    selected_patterns = allow_patterns or _DEFAULT_ALLOW_PATTERNS
    if selected_subdir is not None:
        selected_patterns = [f"{selected_subdir}/**", "README.md"]
    snapshot = Path(
        snapshot_download(
            resolved,
            revision=revision,
            allow_patterns=selected_patterns,
            force_download=force_download,
        )
    )
    if selected_subdir is not None:
        return snapshot.joinpath(*selected_subdir.split("/"))
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
