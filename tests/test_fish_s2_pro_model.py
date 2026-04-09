import mlx.core as mx
import pytest
from mlx_speech.models.fish_s2_pro.llama import DualARTransformer


def test_model_forward():
    """Test basic forward pass."""
    model = DualARTransformer(
        vocab_size=32000,
        num_layers=2,
        dim=512,
        num_heads=4,
    )
    x = mx.array([[1, 2, 3]])
    result = model(x)
    assert result.shape == (1, 3, 32000)


def test_model_structure():
    """Test model has expected components."""
    model = DualARTransformer(num_layers=2, dim=512, num_heads=4)
    assert hasattr(model, "token_embed")
    assert hasattr(model, "layers")
    assert hasattr(model, "norm")
    assert hasattr(model, "lm_head")
