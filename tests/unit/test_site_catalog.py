from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_site_catalog_includes_both_dots_tts_solvers() -> None:
    page = (ROOT / "site/index.html").read_text(encoding="utf-8")

    assert "Fourteen models. One loader." in page
    assert "10 modules" in page
    assert "['dots.tts','SOAR · MeanFlow']" in page
    assert (
        "['dots.tts SOAR','int8 · base','dots-tts-soar','dots-tts.md','dots-tts-mlx'"
    ) in page
    assert (
        "['dots.tts MeanFlow','int8 · base','dots-tts-mf','dots-tts.md','dots-tts-mlx'"
    ) in page


def test_site_catalog_includes_published_nemotron_asr() -> None:
    page = (ROOT / "site/index.html").read_text(encoding="utf-8")

    assert "04 modules" in page
    assert "['nemotron-asr-streaming','cache-aware']" in page
    assert (
        "['Nemotron 3.5 ASR Streaming','int8','nemotron-asr-streaming',"
        "'nemotron-asr.md','nemotron-3.5-asr-streaming-0.6b-int8-mlx'"
    ) in page


def test_readme_and_site_publish_granite_int8_consistently() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "site/index.html").read_text(encoding="utf-8")

    assert "appautomaton/granite-4.0-1b-speech-int8-mlx" in readme
    assert "`granite-speech-4.0-1b`" in readme
    assert "['granite-speech-4.0-1b','selective int8']" in page
    assert (
        "['IBM Granite Speech 4.0 1B','int8 · BF16',"
        "'granite-speech-4.0-1b','granite-speech-asr.md',"
        "'granite-4.0-1b-speech-int8-mlx'"
    ) in page
    assert "session.feed(" in readme
    assert "session.finalize()" in readme
    assert "scripts/convert/dots_tts.py --variant all --precision int8" in readme
    assert "scripts/convert/nemotron_asr.py --quant int8" in readme
    assert "scripts/convert/granite_speech_asr.py" in readme
