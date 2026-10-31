from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.audio8_asr_infinite.config import Audio8Config
from mlx_speech.models.audio8_asr_infinite.text_decoder import (
    Audio8LanguageModel,
    Audio8TimeEmbedding,
)
from mlx_speech.models.audio8_asr_infinite.tokenizer import Audio8Tokenizer
from tests.unit.test_audio8_checkpoint import _config_payload


def test_delay_conditioned_decoder_matches_full_sequence_with_cache():
    mx.random.seed(11)
    config = Audio8Config.from_dict(_config_payload())
    model = Audio8LanguageModel(config.text_config)
    inputs = mx.random.normal((1, 5, 8))
    t_cond = mx.random.normal((1, 1, 8)) * 0.1

    reference = model(inputs_embeds=inputs, t_cond=t_cond, logits_to_keep=0)
    cache = None
    chunks = []
    offset = 0
    for size in (2, 1, 2):
        output = model(
            inputs_embeds=inputs[:, offset : offset + size],
            t_cond=t_cond,
            cache=cache,
            logits_to_keep=0,
        )
        cache = output.cache
        chunks.append(output.logits)
        offset += size
    streamed = mx.concatenate(chunks, axis=1)
    mx.eval(reference.logits, streamed)

    np.testing.assert_allclose(
        np.asarray(streamed), np.asarray(reference.logits), rtol=3e-3, atol=2e-3
    )
    assert cache[0].offset == inputs.shape[1]


def test_time_embedding_uses_delay_tokens_and_requested_dtype():
    embedding = Audio8TimeEmbedding(8)

    output = embedding(6, dtype=mx.bfloat16)

    assert output.shape == (1, 1, 8)
    assert output.dtype == mx.bfloat16
    expected = np.concatenate(
        [np.cos(6 * np.power(10000.0, -np.arange(4) / 4)),
         np.sin(6 * np.power(10000.0, -np.arange(4) / 4))]
    )
    np.testing.assert_allclose(
        np.asarray(output[0, 0].astype(mx.float32)), expected, rtol=2e-2, atol=2e-3
    )
    with pytest.raises(ValueError, match="floating dtype"):
        embedding(6, dtype=mx.uint32)


def test_tokenizer_resolves_streaming_controls_and_visible_text(tmp_path: Path):
    from tokenizers import Tokenizer
    from tokenizers.decoders import ByteLevel
    from tokenizers.models import BPE

    vocabulary = {
        "<|endoftext|>": 0,
        "<|im_start|>": 1,
        "<|im_end|>": 2,
        "[STREAMING_PAD]": 3,
        "[STREAMING_WORD]": 4,
        "[LANGUAGE_ZH]": 5,
        "[LANGUAGE_EN]": 6,
        "你": 7,
        "好": 8,
    }
    tokenizer = Tokenizer(BPE(vocabulary, merges=[]))
    tokenizer.decoder = ByteLevel()
    tokenizer.add_special_tokens(list(vocabulary)[:6])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    payload = _config_payload()
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = Audio8Tokenizer.from_dir(tmp_path)

    assert loaded.language_token_id("zh") == 5
    assert loaded.language_token_id("EN") == 6
    assert loaded.visible_token_ids([3, 7, 4, 8, 2, 7]) == [7, 8]
    assert loaded.decode_visible([3, 7, 4, 8, 2]) == "你好"
