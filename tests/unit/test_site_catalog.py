from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_site_catalog_includes_both_dots_tts_solvers() -> None:
    page = (ROOT / "site/index.html").read_text(encoding="utf-8")

    assert "Thirteen models. One loader." in page
    assert "10 modules" in page
    assert "['dots.tts','SOAR · MeanFlow']" in page
    assert (
        "['dots.tts SOAR','int8 · base','dots-tts-soar','dots-tts.md','dots-tts-mlx'"
    ) in page
    assert (
        "['dots.tts MeanFlow','int8 · base','dots-tts-mf','dots-tts.md','dots-tts-mlx'"
    ) in page
