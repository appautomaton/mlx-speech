from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten

from mlx_speech.models.longcat_audiodit.config import LongCatTextEncoderConfig
from mlx_speech.models.longcat_audiodit.text_encoder import (
    LongCatUMT5Encoder,
    UMT5Attention,
)


def test_tiny_umt5_encoder_returns_expected_shape_and_hidden_states() -> None:
    config = LongCatTextEncoderConfig(
        d_model=16,
        d_ff=32,
        d_kv=4,
        num_heads=4,
        num_layers=2,
        num_decoder_layers=2,
        vocab_size=128,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
    )
    model = LongCatUMT5Encoder(config)

    input_ids = mx.array([[1, 2, 3, 0]], dtype=mx.int32)
    attention_mask = mx.array([[1, 1, 1, 0]], dtype=mx.int32)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )

    assert output.last_hidden_state.shape == (1, 4, 16)
    assert len(output.hidden_states) == 3
    assert output.hidden_states[0].shape == (1, 4, 16)


def test_umt5_parameter_tree_matches_checkpoint_naming() -> None:
    config = LongCatTextEncoderConfig(
        d_model=16,
        d_ff=32,
        d_kv=4,
        num_heads=4,
        num_layers=1,
        num_decoder_layers=1,
        vocab_size=128,
        relative_attention_num_buckets=8,
        relative_attention_max_distance=16,
    )
    model = LongCatUMT5Encoder(config)
    params = tree_flatten(model.parameters(), destination={})

    assert "encoder.embed_tokens.weight" in params
    assert "encoder.block.0.layer.0.SelfAttention.q.weight" in params
    assert (
        "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight" in params
    )
    assert "encoder.block.0.layer.1.DenseReluDense.wi_0.weight" in params
    assert "encoder.block.0.layer.1.DenseReluDense.wi_1.weight" in params
    assert "encoder.block.0.layer.1.DenseReluDense.wo.weight" in params
    assert "encoder.final_layer_norm.weight" in params


def test_umt5_attention_matches_unscaled_t5_scores() -> None:
    config = LongCatTextEncoderConfig(
        d_model=2,
        d_ff=4,
        d_kv=2,
        num_heads=1,
        num_layers=1,
        num_decoder_layers=1,
        vocab_size=8,
        relative_attention_num_buckets=4,
        relative_attention_max_distance=8,
    )
    attention = UMT5Attention(config)
    eye = mx.eye(2, dtype=mx.float32)
    attention.q.weight = eye
    attention.k.weight = eye
    attention.v.weight = eye
    attention.o.weight = eye
    attention.relative_attention_bias.weight = mx.zeros_like(
        attention.relative_attention_bias.weight
    )

    hidden = mx.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=mx.float32)
    mask = mx.array([[1, 1]], dtype=mx.int32)
    output = attention(hidden, mask)

    expected = mx.array(
        [
            [
                [0.7310586, 0.26894143],
                [0.26894143, 0.7310586],
            ]
        ],
        dtype=mx.float32,
    )
    assert mx.allclose(output, expected, atol=2e-4, rtol=2e-4)
