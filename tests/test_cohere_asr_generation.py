from __future__ import annotations

from pathlib import Path

import numpy as np

from mlx_speech.generation.cohere_asr import CohereAsrModel, CohereAsrResult
from mlx_speech.models.cohere_asr.tokenizer import CohereAsrTokenizer


def test_tokenizer_prompt_ids_support_punctuation_and_itn() -> None:
    tokenizer = CohereAsrTokenizer.from_dir(Path("models/cohere/cohere_transcribe/original"))

    default_prompt = tokenizer.get_decoder_prompt_ids("en")
    no_punctuation_prompt = tokenizer.get_decoder_prompt_ids("en", punctuation=False)
    itn_prompt = tokenizer.get_decoder_prompt_ids("en", itn=True)

    assert len(default_prompt) == 10
    assert default_prompt[6] == 5  # <|pnc|>
    assert no_punctuation_prompt[6] == 6  # <|nopnc|>
    assert default_prompt[7] == 9  # <|noitn|>
    assert itn_prompt[7] == 8  # <|itn|>
    assert default_prompt[8:] == [11, 13]  # <|notimestamp|>, <|nodiarize|>
    assert itn_prompt[:7] == default_prompt[:7]


def test_transcribe_batch_preserves_order_and_options(monkeypatch) -> None:
    model = CohereAsrModel(model=None, feature_extractor=None, tokenizer=None, config=None)
    calls: list[tuple[int, int, str, bool, bool, int]] = []

    def fake_transcribe(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        language: str,
        punctuation: bool,
        itn: bool,
        max_new_tokens: int,
    ) -> CohereAsrResult:
        calls.append((len(audio), sample_rate, language, punctuation, itn, max_new_tokens))
        return CohereAsrResult(text=f"len={len(audio)}", tokens=[len(audio)], language=language)

    monkeypatch.setattr(CohereAsrModel, "transcribe", fake_transcribe)

    audios = [np.zeros(8, dtype=np.float32), np.zeros(3, dtype=np.float32)]
    results = model.transcribe_batch(
        audios,
        sample_rate=22050,
        language="fr",
        punctuation=False,
        itn=True,
        max_new_tokens=32,
    )

    assert [result.text for result in results] == ["len=8", "len=3"]
    assert [result.tokens for result in results] == [[8], [3]]
    assert calls == [
        (8, 22050, "fr", False, True, 32),
        (3, 22050, "fr", False, True, 32),
    ]

