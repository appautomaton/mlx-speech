from __future__ import annotations

from pathlib import Path

import pytest

from mlx_speech.models.cohere_asr.tokenizer import CohereAsrTokenizer


MODEL_DIR = Path("models/cohere/cohere_transcribe/original")

pytestmark = [
    pytest.mark.checkpoint,
    pytest.mark.skipif(
        not MODEL_DIR.is_dir(),
        reason="Cohere ASR tokenizer assets are not present",
    ),
]


def test_tokenizer_prompt_ids_support_punctuation_and_itn() -> None:
    tokenizer = CohereAsrTokenizer.from_dir(MODEL_DIR)

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
