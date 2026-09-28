from __future__ import annotations

import mlx.core as mx
import pytest

from mlx_speech.models.breeze_tts.checkpoint import (
    BACKBONE_AUDIO_EMBED,
    DEPTH_AUDIO_EMBED,
    mlx_codec_weight,
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
