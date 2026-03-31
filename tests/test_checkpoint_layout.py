from pathlib import Path

from mlx_voice.checkpoints import get_openmoss_v0_layouts


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
