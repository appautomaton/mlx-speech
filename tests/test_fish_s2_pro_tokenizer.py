import pytest
from mlx_speech.models.fish_s2_pro.tokenizer import FishS2Tokenizer


def test_tokenizer_semantic_tokens():
    """Test semantic token range detection."""
    # Mock tokenizer for testing without actual model files
    tokenizer = FishS2Tokenizer(
        tokenizer=None, semantic_begin_id=256, semantic_end_id=4351
    )
    assert tokenizer.semantic_begin_id == 256
    assert tokenizer.semantic_end_id == 4351
    assert tokenizer.num_semantic_tokens == 4096


def test_tokenizer_properties():
    tokenizer = FishS2Tokenizer(
        tokenizer=None, semantic_begin_id=256, semantic_end_id=4351
    )
    assert tokenizer.vocab_size == 32000  # expected from HF
