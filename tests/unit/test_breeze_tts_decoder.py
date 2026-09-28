from __future__ import annotations

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.breeze_tts.backbone import (
    BreezeBackbone,
    expected_backbone_checkpoint_weights,
)
from mlx_speech.models.breeze_tts.causal import RotaryEmbedding
from mlx_speech.models.breeze_tts.config import (
    BreezeDecoderConfig,
    BreezeRuntimeConfig,
    BreezeTextEncoderConfig,
    RopeScaling,
)
from mlx_speech.models.breeze_tts.depth import (
    BreezeDepthDecoder,
    CodebookHeads,
    expected_depth_checkpoint_weights,
)


def _text() -> BreezeTextEncoderConfig:
    return BreezeTextEncoderConfig.from_runtime(
        {
            "mlx_speech": {"codec": "audio_tokenizer"},
            "hidden_size": 8,
            "text_encoder_config": {
                "vocab_size": 16,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "rms_norm_eps": 1e-6,
                "query_pre_attn_scalar": 4,
                "sliding_window": 4,
                "eoi_token_index": 7,
                "layer_types": ["full_attention"],
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
    )


def _runtime() -> BreezeRuntimeConfig:
    backbone = BreezeDecoderConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=16,
        qk_norm=True,
        cache_growth=4,
    )
    depth = BreezeDecoderConfig(
        hidden_size=4,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        max_position_embeddings=8,
        qk_norm=False,
        rope_scaling=RopeScaling(32.0, 1.0, 4.0, 16),
        cache_growth=4,
    )
    return BreezeRuntimeConfig(
        text_encoder=_text(),
        backbone=backbone,
        depth=depth,
        vocab_size=5,
        num_codebooks=2,
        audio_embed_size=8,
        text_vocab_size=11,
    )


def _load_random(module: object) -> None:
    mx.random.seed(0)
    module.load_weights(  # type: ignore[attr-defined]
        [
            (name, mx.random.normal(value.shape, scale=0.2))
            for name, value in tree_flatten(module.parameters(), destination={}).items()
        ]
    )


def test_backbone_names_match_the_checkpoint_contract() -> None:
    config = _runtime()
    module = BreezeBackbone(config)
    actual = {
        f"backbone_model.{name}": tuple(value.shape)
        for name, value in tree_flatten(module.parameters(), destination={}).items()
    }
    expected = expected_backbone_checkpoint_weights(config)
    assert actual == {
        name: shape
        for name, shape in expected.items()
        if name not in {"lm_head.weight", "embed_text_tokens.weight"}
    }
    assert expected["lm_head.weight"] == (6, 8)
    assert any("self_attn.q_norm.weight" in name for name in actual)


def test_depth_names_omit_qk_norm() -> None:
    config = _runtime()
    module = BreezeDepthDecoder(config)
    actual = {
        f"depth_decoder.{name}": tuple(value.shape)
        for name, value in tree_flatten(module.parameters(), destination={}).items()
    }
    assert actual == expected_depth_checkpoint_weights(config)
    assert all("q_norm" not in name for name in actual)


def test_llama3_rope_scales_only_long_wavelengths() -> None:
    config = BreezeDecoderConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        max_position_embeddings=32,
        qk_norm=False,
        rope_scaling=RopeScaling(8.0, 1.0, 4.0, 8192),
    )
    rope = RotaryEmbedding.from_config(config)
    assert rope.inv_freq[0] == pytest.approx(1.0)
    last_index = config.head_dim - 2
    plain = 1.0 / (config.rope_theta ** (last_index / config.head_dim))
    assert rope.inv_freq[-1] == pytest.approx(plain / config.rope_scaling.factor)


def test_incremental_backbone_matches_prefill() -> None:
    backbone = BreezeBackbone(_runtime())
    _load_random(backbone)
    embeds = mx.random.normal((1, 3, 8), scale=0.3)
    prefill = backbone(embeds, backbone.make_cache(batch_size=1, dtype=embeds.dtype))
    cache = backbone.make_cache(batch_size=1, dtype=embeds.dtype)
    backbone(embeds[:, :2, :], cache)
    stepped = backbone(embeds[:, 2:, :], cache)
    assert cache[0].offset == 3
    assert mx.allclose(prefill[:, -1, :], stepped[:, -1, :], atol=1e-3)


def test_depth_frame_cache_resets() -> None:
    depth = BreezeDepthDecoder(_runtime())
    _load_random(depth)
    cache = depth.make_cache(batch_size=1, dtype=mx.float32)
    backbone_hidden = mx.random.normal((1, 8))
    first_code = mx.random.normal((1, 8))
    depth.begin_frame(backbone_hidden, first_code, cache)
    assert cache[0].offset == 2
    with pytest.raises(ValueError, match="empty cache"):
        depth.begin_frame(backbone_hidden, first_code, cache)
    depth.reset_cache(cache)
    depth.begin_frame(backbone_hidden, first_code, cache)
    assert cache[0].offset == 2


def test_codebook_head_uses_the_previous_position() -> None:
    heads = CodebookHeads(hidden_size=2, num_codebooks=3, vocab_size=4)
    heads.weight = mx.arange(2 * 2 * 4, dtype=mx.float32).reshape(2, 2, 4)
    hidden = mx.ones((1, 1, 2))
    logits = heads(hidden, mx.array([1], dtype=mx.int32))
    assert mx.allclose(logits[0], mx.sum(heads.weight[0], axis=0))


def test_audio_embeddings_sum_codebook_offsets() -> None:
    backbone = BreezeBackbone(_runtime())
    backbone.embed_tokens.embed_audio_tokens.weight = mx.arange(
        10 * 8, dtype=mx.float32
    ).reshape(10, 8)
    codes = mx.array([[[1, 0]]])
    embedded = backbone.embed_audio(codes)
    expected = backbone.embed_tokens.embed_audio_tokens.weight[1]
    expected = expected + backbone.embed_tokens.embed_audio_tokens.weight[5]
    assert mx.allclose(embedded[0, 0], expected)
