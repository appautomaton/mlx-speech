"""Local checkpoint layout helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _resolve_repo_root() -> Path:
    candidate = Path(__file__).resolve().parents[3]
    if candidate.parent.name == ".worktrees":
        shared_root = candidate.parent.parent
        if (shared_root / "models").exists():
            return shared_root
    models_root = candidate / "models"
    if models_root.exists():
        return candidate
    return candidate


REPO_ROOT = _resolve_repo_root()
MODELS_ROOT = REPO_ROOT / "models"


@dataclass(frozen=True)
class ModelArtifactLayout:
    """Filesystem layout for one model family."""

    family: str
    model_name: str
    repo_id: str
    root_dir: Path
    original_dir: Path
    mlx_int8_dir: Path

    def ensure(self) -> "ModelArtifactLayout":
        self.original_dir.mkdir(parents=True, exist_ok=True)
        self.mlx_int8_dir.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class OpenMossV0Layouts:
    """Grouped layout for the v0 OpenMOSS assets."""

    moss_tts_local: ModelArtifactLayout
    audio_tokenizer: ModelArtifactLayout
    moss_sound_effect: ModelArtifactLayout

    def ensure(self) -> "OpenMossV0Layouts":
        self.moss_tts_local.ensure()
        self.audio_tokenizer.ensure()
        self.moss_sound_effect.ensure()
        return self


@dataclass(frozen=True)
class StepFunV4Layouts:
    """Grouped layout for the Step-Audio-EditX family."""

    step_audio_editx: ModelArtifactLayout
    step_audio_tokenizer: ModelArtifactLayout

    def ensure(self) -> "StepFunV4Layouts":
        self.step_audio_editx.ensure()
        self.step_audio_tokenizer.ensure()
        return self


def _build_model_layout(
    models_root: Path,
    family: str,
    model_name: str,
    repo_id: str,
) -> ModelArtifactLayout:
    root_dir = models_root / family / model_name
    return ModelArtifactLayout(
        family=family,
        model_name=model_name,
        repo_id=repo_id,
        root_dir=root_dir,
        original_dir=root_dir / "original",
        mlx_int8_dir=root_dir / "mlx-int8",
    )


def get_openmoss_v0_layouts(models_root: Path | None = None) -> OpenMossV0Layouts:
    """Return the local model layout used by the current v0 plan."""

    resolved_root = MODELS_ROOT if models_root is None else Path(models_root)
    return OpenMossV0Layouts(
        moss_tts_local=_build_model_layout(
            models_root=resolved_root,
            family="openmoss",
            model_name="moss_tts_local",
            repo_id="OpenMOSS-Team/MOSS-TTS-Local-Transformer",
        ),
        audio_tokenizer=_build_model_layout(
            models_root=resolved_root,
            family="openmoss",
            model_name="moss_audio_tokenizer",
            repo_id="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        ),
        moss_sound_effect=_build_model_layout(
            models_root=resolved_root,
            family="openmoss",
            model_name="moss_sound_effect",
            repo_id="OpenMOSS-Team/MOSS-SoundEffect",
        ),
    )


def get_stepfun_v4_layouts(models_root: Path | None = None) -> StepFunV4Layouts:
    """Return the local model layout used by the Step-Audio v4 plan."""

    resolved_root = MODELS_ROOT if models_root is None else Path(models_root)
    return StepFunV4Layouts(
        step_audio_editx=_build_model_layout(
            models_root=resolved_root,
            family="stepfun",
            model_name="step_audio_editx",
            repo_id="stepfun-ai/Step-Audio-EditX",
        ),
        step_audio_tokenizer=_build_model_layout(
            models_root=resolved_root,
            family="stepfun",
            model_name="step_audio_tokenizer",
            repo_id="stepfun-ai/Step-Audio-Tokenizer",
        ),
    )
