from pathlib import Path

from mlx_speech.checkpoints import get_openmoss_v0_layouts
from mlx_speech.checkpoints import layout as layout_module


def test_openmoss_v0_layouts_are_built_under_models_root(tmp_path: Path) -> None:
    layouts = get_openmoss_v0_layouts(models_root=tmp_path)

    assert layouts.moss_tts_local.original_dir == (
        tmp_path / "openmoss" / "moss_tts_local" / "original"
    )
    assert layouts.moss_tts_local.mlx_int8_dir == (
        tmp_path / "openmoss" / "moss_tts_local" / "mlx-int8"
    )
    assert layouts.audio_tokenizer.original_dir == (
        tmp_path / "openmoss" / "moss_audio_tokenizer" / "original"
    )
    assert layouts.audio_tokenizer.mlx_int8_dir == (
        tmp_path / "openmoss" / "moss_audio_tokenizer" / "mlx-int8"
    )
    assert layouts.moss_sound_effect.original_dir == (
        tmp_path / "openmoss" / "moss_sound_effect" / "original"
    )
    assert layouts.moss_sound_effect.mlx_int8_dir == (
        tmp_path / "openmoss" / "moss_sound_effect" / "mlx-int8"
    )


def test_openmoss_v0_layouts_can_create_directories(tmp_path: Path) -> None:
    layouts = get_openmoss_v0_layouts(models_root=tmp_path).ensure()

    assert layouts.moss_tts_local.original_dir.is_dir()
    assert layouts.moss_tts_local.mlx_int8_dir.is_dir()
    assert layouts.audio_tokenizer.original_dir.is_dir()
    assert layouts.audio_tokenizer.mlx_int8_dir.is_dir()
    assert layouts.moss_sound_effect.original_dir.is_dir()
    assert layouts.moss_sound_effect.mlx_int8_dir.is_dir()


def test_resolve_checkout_root_uses_current_checkout_root(tmp_path: Path) -> None:
    checkout_root = tmp_path / "repo"
    source_file = checkout_root / "src" / "mlx_speech" / "checkpoints" / "layout.py"
    source_file.parent.mkdir(parents=True)
    source_file.touch()

    assert layout_module._resolve_checkout_root(source_file) == checkout_root.resolve()


def test_resolve_models_root_uses_shared_parent_for_worktrees(tmp_path: Path, monkeypatch) -> None:
    shared_root = tmp_path / "repo"
    shared_models = shared_root / "models"
    shared_models.mkdir(parents=True)
    worktree_root = shared_root / ".worktrees" / "feature-a"
    worktree_root.mkdir(parents=True)
    monkeypatch.delenv("MLX_SPEECH_MODELS_ROOT", raising=False)

    assert layout_module._resolve_models_root_for_checkout(worktree_root) == shared_models.resolve()


def test_resolve_models_root_allows_explicit_override(tmp_path: Path, monkeypatch) -> None:
    checkout_root = tmp_path / "repo"
    override = tmp_path / "custom-models"
    override.mkdir(parents=True)
    monkeypatch.setenv("MLX_SPEECH_MODELS_ROOT", str(override))

    assert layout_module._resolve_models_root_for_checkout(checkout_root) == override.resolve()


def test_resolve_models_root_falls_back_to_checkout_models(tmp_path: Path, monkeypatch) -> None:
    checkout_root = tmp_path / "repo"
    checkout_models = checkout_root / "models"
    checkout_models.mkdir(parents=True)
    monkeypatch.delenv("MLX_SPEECH_MODELS_ROOT", raising=False)

    assert layout_module._resolve_models_root_for_checkout(checkout_root) == checkout_models.resolve()
