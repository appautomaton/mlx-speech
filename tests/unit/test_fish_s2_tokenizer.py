from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from mlx_speech.models.fish_s2_pro.tokenizer import (
    IM_END_TOKEN,
    IM_START_TOKEN,
    MODALITY_TOKENS,
    FishS2Tokenizer,
)


def _write_tokenizer_dir(
    tmp_path: Path,
    *,
    vocab_overrides: dict[str, int] | None = None,
    config_text: str = '{"eos_token":"<|im_end|>","pad_token":"[UNK]"}',
) -> Path:
    vocab = {
        "[UNK]": 0,
        IM_START_TOKEN: 10,
        IM_END_TOKEN: 11,
        "<|text|>": 12,
        "<|voice|>": 13,
        "<|interleave|>": 14,
        "user\n": 15,
        "hello": 16,
        "<|semantic:0|>": 151678,
        "<|semantic:1|>": 151700,
    }
    if vocab_overrides:
        for token, token_id in vocab_overrides.items():
            if token_id is None:
                vocab.pop(token, None)
            else:
                vocab[token] = token_id
    tok = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(config_text, encoding="utf-8")
    return tmp_path


class _FakeTokenizer:
    vocab_size = 200000
    pad_token_id = 1
    eos_token_id = 2

    def __init__(self):
        self._vocab = {
            IM_START_TOKEN: 10,
            IM_END_TOKEN: 11,
            "<|text|>": 12,
            "<|voice|>": 13,
            "<|interleave|>": 14,
            "<|semantic:0|>": 151678,
            "<|semantic:1|>": 151700,
        }

    def get_vocab(self):
        return self._vocab

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [self._vocab.get(text, 99)]

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]


def test_tokenizer_discovers_semantic_range():
    tok = FishS2Tokenizer(_FakeTokenizer())

    assert tok.semantic_begin_id == 151678
    assert tok.semantic_end_id == 151700


def test_tokenizer_exposes_control_token_helpers():
    tok = FishS2Tokenizer(_FakeTokenizer())

    assert tok.im_start_id == 10
    assert tok.im_end_id == 11
    assert tok.modality_id("voice") == 13
    assert tok.modality_id("interleave") == 14
    assert tok.semantic_id(0) == 151678
    assert tok.semantic_id(1) == 151700


def test_tokenizer_raises_for_unknown_semantic_code():
    tok = FishS2Tokenizer(_FakeTokenizer())

    with pytest.raises(ValueError, match="Semantic token code 2 is not available"):
        tok.semantic_id(2)


def test_from_pretrained_reads_tokenizer_json_without_transformers(tmp_path):
    model_dir = _write_tokenizer_dir(tmp_path)

    tok = FishS2Tokenizer.from_pretrained(model_dir)

    assert tok.im_start_id == 10
    assert tok.im_end_id == 11
    assert tok.semantic_id(0) == 151678
    assert tok.semantic_id(1) == 151700


def test_from_pretrained_raises_for_missing_tokenizer_json(tmp_path):
    with pytest.raises(FileNotFoundError, match="tokenizer.json"):
        FishS2Tokenizer.from_pretrained(tmp_path)


def test_from_pretrained_rejects_sparse_semantic_ids(tmp_path):
    model_dir = _write_tokenizer_dir(
        tmp_path,
        vocab_overrides={"<|semantic:1|>": None, "<|semantic:2|>": 151701},
    )

    with pytest.raises(ValueError, match="dense sequence"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_non_dict_tokenizer_config(tmp_path):
    model_dir = _write_tokenizer_dir(tmp_path, config_text='["not", "an", "object"]')

    with pytest.raises(ValueError, match="tokenizer_config.json.*JSON object"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_malformed_tokenizer_config(tmp_path):
    model_dir = _write_tokenizer_dir(tmp_path, config_text="{")

    with pytest.raises(ValueError, match="Invalid Fish tokenizer_config.json"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_missing_control_token(tmp_path):
    model_dir = _write_tokenizer_dir(tmp_path, vocab_overrides={IM_START_TOKEN: None})

    with pytest.raises(ValueError, match=IM_START_TOKEN):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_missing_modality_token(tmp_path):
    model_dir = _write_tokenizer_dir(
        tmp_path,
        vocab_overrides={MODALITY_TOKENS["interleave"]: None},
    )

    with pytest.raises(ValueError, match=MODALITY_TOKENS["interleave"]):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_missing_semantic_tokens(tmp_path):
    model_dir = _write_tokenizer_dir(
        tmp_path,
        vocab_overrides={"<|semantic:0|>": None, "<|semantic:1|>": None},
    )

    with pytest.raises(ValueError, match="No semantic tokens"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_malformed_tokenizer_json(tmp_path):
    model_dir = _write_tokenizer_dir(tmp_path)
    (model_dir / "tokenizer.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid Fish tokenizer.json"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_invalid_pad_token_config_entry(tmp_path):
    model_dir = _write_tokenizer_dir(
        tmp_path,
        config_text='{"eos_token":"<|im_end|>","pad_token":{"id":0}}',
    )

    with pytest.raises(ValueError, match="pad_token"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_unknown_pad_token_config_entry(tmp_path):
    model_dir = _write_tokenizer_dir(
        tmp_path,
        config_text='{"eos_token":"<|im_end|>","pad_token":"<|missing_pad|>"}',
    )

    with pytest.raises(ValueError, match="pad_token.*<\|missing_pad\|>"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_rejects_invalid_eos_token_config_entry(tmp_path):
    model_dir = _write_tokenizer_dir(
        tmp_path,
        config_text='{"eos_token":[],"pad_token":"[UNK]"}',
    )

    with pytest.raises(ValueError, match="eos_token"):
        FishS2Tokenizer.from_pretrained(model_dir)


def test_from_pretrained_accepts_structured_special_token_config_entries(tmp_path):
    model_dir = _write_tokenizer_dir(
        tmp_path,
        config_text=(
            '{"eos_token":{"content":"<|im_end|>"},"pad_token":{"content":"[UNK]"}}'
        ),
    )

    tok = FishS2Tokenizer.from_pretrained(model_dir)

    assert tok._tokenizer.pad_token_id == 0
    assert tok._tokenizer.eos_token_id == 11
