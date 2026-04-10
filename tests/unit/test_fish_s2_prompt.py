import mlx.core as mx
import pytest

from mlx_speech.models.fish_s2_pro.prompt import (
    Conversation,
    Message,
    TextPart,
    VQPart,
)
from mlx_speech.models.fish_s2_pro.tokenizer import (
    IM_END_TOKEN,
    IM_START_TOKEN,
    FishS2Tokenizer,
)


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
            "user\n": 20,
            "system\n": 21,
            "assistant\n": 22,
            "hello": 30,
            "<|semantic:0|>": 151678,
            "<|semantic:1|>": 151700,
        }

    def get_vocab(self):
        return self._vocab

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [self._vocab.get(text, 99)]

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]


def test_conversation_encodes_interleaved_prompt():
    tok = FishS2Tokenizer(_FakeTokenizer())
    convo = Conversation()
    convo.append(
        Message(
            role="user",
            modality="voice",
            parts=[TextPart("hello"), VQPart([[0, 1]])],
        )
    )

    encoded = convo.encode_for_inference(tok, num_codebooks=1)

    assert encoded.shape[0] == 2
    assert encoded.shape[1] == 8
    assert encoded[0].tolist() == [10, 20, 13, 30, 151678, 151700, 11, 99]
    assert encoded[1].tolist() == [0, 0, 0, 0, 0, 1, 0, 0]


def test_conversation_rejects_empty_messages():
    tok = FishS2Tokenizer(_FakeTokenizer())

    with pytest.raises(ValueError, match="Conversation produced no prompt tokens"):
        Conversation().encode_for_inference(tok, num_codebooks=1)


def test_conversation_rejects_non_2d_vq_codes():
    tok = FishS2Tokenizer(_FakeTokenizer())
    convo = Conversation(messages=[Message(role="user", parts=[VQPart([0, 1])])])

    with pytest.raises(ValueError, match="VQ parts must be a 2-D array"):
        convo.encode_for_inference(tok, num_codebooks=1)


def test_conversation_rejects_empty_vq_codes():
    tok = FishS2Tokenizer(_FakeTokenizer())
    convo = Conversation(messages=[Message(role="user", parts=[VQPart([[]])])])

    with pytest.raises(ValueError, match="VQ parts must be non-empty"):
        convo.encode_for_inference(tok, num_codebooks=1)


def test_conversation_rejects_codebook_count_mismatch():
    tok = FishS2Tokenizer(_FakeTokenizer())
    convo = Conversation(messages=[Message(role="user", parts=[VQPart([[0, 1]])])])

    with pytest.raises(ValueError, match="VQ parts must have exactly 2 codebook rows"):
        convo.encode_for_inference(tok, num_codebooks=2)
