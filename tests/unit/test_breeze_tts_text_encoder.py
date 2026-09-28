from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.breeze_tts.config import BreezeTextEncoderConfig
from mlx_speech.models.breeze_tts.text_encoder import (
    BreezeTextEncoder,
    expected_text_checkpoint_weights,
    gemma_rms_norm,
    text_attention_mask,
)


def _config(**overrides: object) -> BreezeTextEncoderConfig:
    payload = {
        "mlx_speech": {"codec": "audio_tokenizer"},
        "hidden_size": 32,
        "text_encoder_config": {
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 1e-6,
            "query_pre_attn_scalar": 16,
            "sliding_window": 4,
            "eoi_token_index": 7,
            "layer_types": ["sliding_attention", "full_attention"],
            "rope_parameters": {
                "full_attention": {
                    "rope_type": "linear",
                    "rope_theta": 1000.0,
                    "factor": 8.0,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10.0},
            },
            "hidden_activation": "gelu_pytorch_tanh",
            "attn_logit_softcapping": None,
            "attention_bias": False,
        },
    }
    text = payload["text_encoder_config"]
    assert isinstance(text, dict)
    text.update(overrides)
    return BreezeTextEncoderConfig.from_runtime(payload)


def test_runtime_config_rejects_a_mimi_codec_selection() -> None:
    with pytest.raises(ValueError, match="audio_tokenizer"):
        BreezeTextEncoderConfig.from_runtime(
            {
                "mlx_speech": {"codec": "mimi"},
                "hidden_size": 8,
                "text_encoder_config": {},
            }
        )


def test_text_config_rejects_softcapping() -> None:
    with pytest.raises(ValueError, match="soft-capping"):
        _config(attn_logit_softcapping=50.0)


def test_gemma_rms_norm_uses_the_weight_as_an_offset() -> None:
    value = mx.array([[3.0, 4.0]])
    weight = mx.array([1.0, -0.5])
    output = gemma_rms_norm(value, weight, 1e-6)
    manual = mx.array([[3.0, 4.0]]) / mx.sqrt(mx.mean(mx.array([9.0, 16.0])))
    assert mx.allclose(output, manual * mx.array([2.0, 0.5]))


def test_sliding_mask_looks_behind_and_ahead() -> None:
    mask = text_attention_mask(
        None,
        batch_size=1,
        sequence_length=8,
        layer_type="sliding_attention",
        sliding_window=4,
        dtype=mx.float32,
    )
    assert mask is not None
    visible = mask[0, 0, 5] == 0
    # Window 4: one position back, self, and two positions ahead.
    assert visible.tolist() == [False, False, False, False, True, True, True, True]


def test_full_attention_mask_is_absent_without_padding() -> None:
    assert (
        text_attention_mask(
            None,
            batch_size=2,
            sequence_length=3,
            layer_type="full_attention",
            sliding_window=4,
            dtype=mx.float32,
        )
        is None
    )


def test_full_attention_mask_hides_padding() -> None:
    mask = text_attention_mask(
        mx.array([[1, 1, 0]]),
        batch_size=1,
        sequence_length=3,
        layer_type="full_attention",
        sliding_window=4,
        dtype=mx.float32,
    )
    assert mask is not None
    assert (mask[0, 0, 0] == 0).tolist() == [True, True, False]


def test_attention_scale_uses_query_pre_attn_scalar() -> None:
    encoder = BreezeTextEncoder(_config())
    assert encoder.layers[0].self_attn.scale == pytest.approx(16.0**-0.5)


def test_linear_rope_divides_inverse_frequency() -> None:
    encoder = BreezeTextEncoder(_config())
    full = encoder.layers[1].self_attn.rope._inv_freq
    sliding = encoder.layers[0].self_attn.rope._inv_freq
    assert full[0] == pytest.approx(sliding[0] / 8.0)


def test_eoi_token_replaces_the_scaled_word_embedding() -> None:
    encoder = BreezeTextEncoder(_config())
    encoder.embed_tokens.weight = mx.ones_like(encoder.embed_tokens.weight)
    encoder.embed_tokens.eoi_embedding = mx.full((8,), 3.0)
    output = encoder.embed_tokens(mx.array([[1, 7]]))
    scale = 8.0**0.5
    assert mx.allclose(output[0, 0], mx.full((8,), scale))
    assert mx.allclose(output[0, 1], mx.full((8,), 3.0))


def test_text_encoder_parameter_names_match_the_checkpoint_contract() -> None:
    config = _config()
    encoder = BreezeTextEncoder(config)
    expected = expected_text_checkpoint_weights(config)
    actual = {
        f"text_encoder.{name}": tuple(value.shape)
        for name, value in tree_flatten(encoder.parameters(), destination={}).items()
    }
    assert actual == {
        name: shape
        for name, shape in expected.items()
        if name != "text_encoder_proj.weight"
    }
    assert expected["text_encoder_proj.weight"] == (32, 8)


def test_tiny_text_encoder_respects_padding() -> None:
    encoder = BreezeTextEncoder(_config())
    mx.random.seed(0)
    encoder.load_weights(
        [
            (name, mx.random.normal(value.shape, scale=0.2))
            for name, value in tree_flatten(
                encoder.parameters(), destination={}
            ).items()
        ]
    )
    tokens = mx.array([[1, 2, 3, 4]])
    unpadded = encoder(tokens, attention_mask=mx.ones((1, 4)))
    padded = encoder(tokens, attention_mask=mx.array([[1, 1, 1, 0]]))
    assert unpadded.shape == (1, 4, 8)
    assert mx.all(mx.isfinite(unpadded))
    assert not mx.allclose(unpadded, padded)
