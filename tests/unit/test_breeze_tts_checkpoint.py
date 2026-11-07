from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_speech.models.breeze_tts.checkpoint import (
    BACKBONE_AUDIO_EMBED,
    DEPTH_AUDIO_EMBED,
    BreezeQuantization,
    mlx_codec_weight,
    quantization_paths,
    quantize_main_weights,
    select_main_weights,
)


def test_main_weights_drop_mimi_and_materialize_the_tied_embedding() -> None:
    depth = mx.ones((4, 2), dtype=mx.bfloat16)
    selected = select_main_weights(
        {
            "codec_model.decoder.weight": mx.zeros((2, 2), dtype=mx.bfloat16),
            DEPTH_AUDIO_EMBED: depth,
            "lm_head.weight": mx.ones((3, 2), dtype=mx.bfloat16),
        }
    )

    assert "codec_model.decoder.weight" not in selected
    assert selected[BACKBONE_AUDIO_EMBED] is depth
    assert selected["lm_head.weight"].dtype == mx.bfloat16


def test_main_weights_reject_a_non_bf16_tensor() -> None:
    with pytest.raises(ValueError, match="bfloat16"):
        select_main_weights(
            {
                DEPTH_AUDIO_EMBED: mx.ones((2, 2), dtype=mx.float32),
            }
        )


def test_codec_convolution_layouts() -> None:
    regular, regular_changed = mlx_codec_weight(
        "decoder.pre_conv.conv.weight",
        mx.arange(24, dtype=mx.float32).reshape(4, 2, 3),
    )
    transposed, transposed_changed = mlx_codec_weight(
        "decoder.decoder.1.block.1.conv.weight",
        mx.arange(24, dtype=mx.float32).reshape(4, 2, 3),
    )
    upsample, upsample_changed = mlx_codec_weight(
        "decoder.upsample.0.0.conv.weight",
        mx.arange(24, dtype=mx.float32).reshape(4, 2, 3),
    )
    depthwise, depthwise_changed = mlx_codec_weight(
        "decoder.upsample.0.1.dwconv.conv.weight",
        mx.arange(21, dtype=mx.float32).reshape(3, 1, 7),
    )
    projection, projection_changed = mlx_codec_weight(
        "encoder.quantizer.semantic_residual_vector_quantizer.input_proj.weight",
        mx.arange(8, dtype=mx.float32).reshape(2, 4, 1),
    )
    bias, bias_changed = mlx_codec_weight(
        "decoder.pre_conv.conv.bias",
        mx.ones((4,), dtype=mx.float32),
    )

    assert regular_changed and tuple(regular.shape) == (4, 3, 2)
    assert transposed_changed and tuple(transposed.shape) == (2, 3, 4)
    assert upsample_changed and tuple(upsample.shape) == (2, 3, 4)
    assert depthwise_changed and tuple(depthwise.shape) == (3, 7, 1)
    assert projection_changed and tuple(projection.shape) == (2, 1, 4)
    assert not bias_changed and tuple(bias.shape) == (4,)
    assert regular.dtype == mx.float32


def test_codec_weight_rejects_an_unknown_rank3_tensor() -> None:
    with pytest.raises(ValueError, match="rank-3"):
        mlx_codec_weight(
            "decoder.codebook.weight",
            mx.ones((2, 3, 4), dtype=mx.float32),
        )


def test_int8_targets_only_the_layer_projections() -> None:
    names = [
        "backbone_model.layers.0.self_attn.q_proj.weight",
        "backbone_model.layers.27.mlp.down_proj.weight",
        "depth_decoder.model.layers.3.self_attn.o_proj.weight",
        "depth_decoder.model.inputs_embeds_projector.weight",
        "depth_decoder.codebooks_head.weight",
        "lm_head.weight",
        "text_encoder.layers.0.self_attn.q_proj.weight",
        "backbone_model.layers.0.input_layernorm.weight",
        "embed_text_tokens.weight",
    ]
    assert quantization_paths(names) == (
        "backbone_model.layers.0.self_attn.q_proj",
        "backbone_model.layers.27.mlp.down_proj",
        "depth_decoder.model.layers.3.self_attn.o_proj",
    )


def test_int8_packs_targets_and_leaves_the_rest() -> None:
    target = "backbone_model.layers.0.mlp.up_proj"
    head = mx.ones((4, 64), dtype=mx.bfloat16)
    packed, quantization = quantize_main_weights(
        {f"{target}.weight": mx.ones((8, 64), dtype=mx.bfloat16), "lm_head.weight": head}
    )
    assert quantization.paths == (target,)
    assert packed[f"{target}.weight"].dtype == mx.uint32
    assert {f"{target}.scales", f"{target}.biases"} <= packed.keys()
    assert packed["lm_head.weight"] is head
    assert BreezeQuantization.from_dict(quantization.to_dict()) == quantization


def test_int8_metadata_rejects_a_path_outside_the_layers() -> None:
    with pytest.raises(ValueError, match="may not quantize"):
        BreezeQuantization.from_dict(
            {"bits": 8, "group_size": 64, "mode": "affine", "paths": ["lm_head"]}
        )
    with pytest.raises(ValueError, match="8-bit"):
        BreezeQuantization.from_dict(
            {"bits": 4, "group_size": 64, "mode": "affine", "paths": ["backbone_model.layers.0.mlp.up_proj"]}
        )
