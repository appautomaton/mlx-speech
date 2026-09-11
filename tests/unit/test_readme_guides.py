from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_readme_and_model_guides_publish_supported_checkpoints() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "appautomaton/granite-4.0-1b-speech-int8-mlx" in readme
    assert "`granite-speech-4.0-1b`" in readme
    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert 'write_wav("output.wav", result.waveform' in readme
    assert "local-only adapters" not in readme
    assert "session.feed(" in readme
    assert "session.finalize()" in readme
    for guide_name, converter, precision_flag in (
        ("dots-tts.md", "dots_tts.py", "--precision int8"),
        ("nemotron-asr.md", "nemotron_asr.py", "--quant int8"),
        ("granite-speech-asr.md", "granite_speech_asr.py", None),
    ):
        assert f"/docs/{guide_name}" in readme
        guide = (ROOT / "docs" / guide_name).read_text(encoding="utf-8")
        assert f"scripts/convert/{converter}" in guide
        if precision_flag is not None:
            assert precision_flag in guide
