from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).parents[2]


class _CatalogParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.cards: dict[str, dict] = {}
        self.card: dict | None = None
        self.in_heading = False
        self.link: dict | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs = dict(attrs)
        if tag == "article" and "data-alias" in attrs:
            alias = attrs["data-alias"]
            assert alias not in self.cards
            self.card = {
                "heading": "",
                "text": "",
                "links": [],
                "asr": attrs["data-task"] == "asr",
            }
            self.cards[alias] = self.card
        if self.card is not None:
            if tag == "h3":
                self.in_heading = True
            if tag == "a":
                self.link = {**attrs, "text": ""}
                self.card["links"].append(self.link)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            self.link = None
        if tag == "h3":
            self.in_heading = False
        if tag == "article":
            self.card = None

    def handle_data(self, data: str) -> None:
        if self.card is not None:
            self.card["text"] += data
            if self.in_heading:
                self.card["heading"] += data
            if self.link is not None:
                self.link["text"] += data


def _catalog() -> dict[str, dict]:
    parser = _CatalogParser()
    parser.feed((ROOT / "site/index.html").read_text(encoding="utf-8"))
    return parser.cards


def _weights(card: dict) -> str:
    links = [
        link["href"] for link in card["links"]
        if link["href"].startswith("https://huggingface.co/appautomaton/")
    ]
    assert len(links) == 1
    return links[0]


def test_site_catalog_has_crawlable_model_headings_and_links() -> None:
    cards = _catalog()
    assert len(cards) == 15
    assert sum(card["asr"] for card in cards.values()) == 4
    for alias, card in cards.items():
        assert card["heading"], alias
        assert alias in card["text"]
        guides = {
            link["href"] for link in card["links"]
            if link["href"].startswith(
                "https://github.com/appautomaton/mlx-speech/blob/main/docs/"
            )
        }
        assert len(guides) == 1
        assert (ROOT / "docs" / next(iter(guides)).rsplit("/", 1)[-1]).is_file()
        assert _weights(card)
        for link in card["links"]:
            assert "nofollow" not in link.get("rel", "").split()
            assert card["heading"] in link.get("aria-label", link["text"])


def test_site_catalog_includes_both_dots_tts_solvers() -> None:
    cards = _catalog()

    assert sum(not card["asr"] for card in cards.values()) == 11
    assert "10-step flow-matching solver" in cards["dots-tts-soar"]["text"]
    assert "four-step distilled acoustic solver" in cards["dots-tts-mf"]["text"]
    for alias, heading in (
        ("dots-tts-soar", "dots.tts SOAR"),
        ("dots-tts-mf", "dots.tts MeanFlow"),
    ):
        assert cards[alias]["heading"] == heading
        assert (
            _weights(cards[alias])
            == "https://huggingface.co/appautomaton/dots-tts-mlx"
        )


def test_site_catalog_includes_published_nemotron_asr() -> None:
    card = _catalog()["nemotron-asr-streaming"]
    assert card["asr"]
    assert "Cache-aware multilingual streaming ASR" in card["text"]
    assert card["heading"] == "Nemotron 3.5 ASR Streaming"
    assert (
        _weights(card)
        == "https://huggingface.co/appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx"
    )


def test_site_fireredtts3_points_to_base_bf16_with_install_requirement() -> None:
    card = _catalog()["fireredtts3-base"]
    assert card["heading"] == "FireRedTTS3 Base"
    assert "mono 24\N{NO-BREAK SPACE}kHz" in card["text"]
    assert "Requires the current GitHub install" in card["text"]
    assert (
        _weights(card)
        == "https://huggingface.co/appautomaton/fireredtts3-mlx/tree/main/base/mlx-bf16"
    )


def test_readme_and_site_publish_granite_int8_consistently() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "appautomaton/granite-4.0-1b-speech-int8-mlx" in readme
    assert "`granite-speech-4.0-1b`" in readme
    assert "actions/workflows/ci.yml/badge.svg" in readme
    assert 'write_wav("output.wav", result.waveform' in readme
    assert "local-only adapters" not in readme
    card = _catalog()["granite-speech-4.0-1b"]
    assert card["asr"]
    assert "Selective-int8 Granite LM" in card["text"]
    assert card["heading"] == "IBM Granite Speech 4.0 1B"
    assert "int8 · BF16" in card["text"]
    assert (
        _weights(card)
        == "https://huggingface.co/appautomaton/granite-4.0-1b-speech-int8-mlx"
    )
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
