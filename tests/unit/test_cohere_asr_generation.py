from __future__ import annotations

import numpy as np

from mlx_speech.generation.cohere_asr import CohereAsrModel, CohereAsrResult


def test_transcribe_batch_preserves_order_and_options(monkeypatch) -> None:
    model = CohereAsrModel(
        model=None, feature_extractor=None, tokenizer=None, config=None
    )
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
        calls.append(
            (len(audio), sample_rate, language, punctuation, itn, max_new_tokens)
        )
        return CohereAsrResult(
            text=f"len={len(audio)}", tokens=[len(audio)], language=language
        )

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
