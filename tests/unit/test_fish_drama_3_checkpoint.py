import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.fish_drama_3.checkpoint import (
    dequantize_fp8_blocks,
    remap_key,
    to_mlx_dense,
)


def decode_e4m3fn(bits: np.ndarray) -> np.ndarray:
    """float8_e4m3fn bits -> float64, straight from the OCP FP8 definition."""
    bits = bits.astype(np.int64)
    sign = np.where(bits >> 7, -1.0, 1.0)
    exponent, mantissa = (bits >> 3) & 0xF, bits & 0x7
    magnitude = np.where(
        exponent == 0,
        mantissa / 8 * 2.0**-6,
        (1 + mantissa / 8) * 2.0 ** (exponent - 7),
    )
    return np.where((exponent == 0xF) & (mantissa == 0x7), np.nan, sign * magnitude)


def random_fp8_bits(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    bits = rng.integers(0, 256, size=shape, dtype=np.uint8)
    return np.where((bits & 0x7F) == 0x7F, bits ^ 0x01, bits).astype(np.uint8)


def block_oracle(bits: np.ndarray, scales: np.ndarray) -> np.ndarray:
    expanded = np.repeat(np.repeat(scales.astype(np.float64), 128, -2), 128, -1)
    return decode_e4m3fn(bits) * expanded


def test_every_finite_fp8_code_decodes_to_its_value():
    bits = np.array([c for c in range(256) if c & 0x7F != 0x7F], dtype=np.uint8)
    weight = np.tile(bits, (128, 1))[:, : 2 * 128 - 2]
    weight = np.pad(weight, ((0, 0), (0, 2)))
    scales = np.ones((1, 2), dtype=np.float32)

    got = dequantize_fp8_blocks(mx.array(weight), mx.array(scales))

    np.testing.assert_array_equal(np.array(got), block_oracle(weight, scales))


def test_block_scales_apply_per_128_tile_across_stacked_experts():
    rng = np.random.default_rng(0)
    bits = random_fp8_bits(rng, (3, 256, 384))
    scales = rng.uniform(1e-4, 2e-3, size=(3, 2, 3)).astype(np.float32)

    got = np.array(dequantize_fp8_blocks(mx.array(bits), mx.array(scales)))

    expected = block_oracle(bits, scales)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=2.0**-24, atol=0)


@pytest.mark.parametrize("nan_bits", [0x7F, 0xFF])
def test_rejects_fp8_nan_codes(nan_bits):
    weight = np.zeros((128, 128), dtype=np.uint8)
    weight[5, 7] = nan_bits

    with pytest.raises(ValueError, match="NaN"):
        dequantize_fp8_blocks(mx.array(weight), mx.ones((1, 1)))


def test_rejects_scales_that_do_not_tile_the_weight():
    weight = mx.zeros((256, 128), dtype=mx.uint8)

    with pytest.raises(ValueError, match="does not tile"):
        dequantize_fp8_blocks(weight, mx.ones((1, 1)))
    with pytest.raises(ValueError, match="not tiled"):
        dequantize_fp8_blocks(mx.zeros((100, 128), dtype=mx.uint8), mx.ones((1, 1)))
    with pytest.raises(TypeError, match="uint8"):
        dequantize_fp8_blocks(mx.zeros((128, 128)), mx.ones((1, 1)))


@pytest.mark.parametrize(
    ("upstream", "mlx"),
    [
        ("model.language_model.embed_tokens.weight", "backbone.embed_tokens.weight"),
        (
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "backbone.layers.3.self_attn.q_proj.weight",
        ),
        (
            "model.language_model.layers.0.mlp.shared_expert.up_proj.weight",
            "backbone.layers.0.mlp.shared_expert.up_proj.weight",
        ),
        ("model.language_model.norm.weight", "backbone.norm.weight"),
        ("lm_head.weight", "backbone.lm_head.weight"),
        (
            "audio_decoder.codebook_embeddings.weight",
            "audio_decoder.codebook_embeddings.weight",
        ),
        (
            "audio_decoder_mtp_shared.layers.2.semantic_embedding.weight",
            "audio_decoder.shared_layers.2.semantic_embedding.weight",
        ),
        (
            "audio_decoder_mtp.layers.8.transformer_layer.attention.wqkv.weight",
            "audio_decoder.residual_layers.8.transformer_layer.attention.wqkv.weight",
        ),
        ("audio_decoder_mtp.heads.0.weight", "audio_decoder.codebook_heads.0.weight"),
    ],
)
def test_remaps_upstream_keys(upstream, mlx):
    assert remap_key(upstream) == mlx


def test_remap_refuses_experts_and_unknown_keys():
    with pytest.raises(ValueError, match="routed expert"):
        remap_key("model.language_model.layers.0.mlp.experts.7.down_proj.weight")
    with pytest.raises(ValueError, match="unknown"):
        remap_key("vision_tower.patch_embed.weight")


def test_depthwise_conv_moves_kernel_to_the_middle_axis():
    torch_layout = mx.arange(6 * 4).reshape(6, 1, 4)

    name, value = to_mlx_dense(
        "model.language_model.layers.0.linear_attn.conv1d.weight", torch_layout
    )

    assert name == "backbone.layers.0.linear_attn.conv1d.weight"
    assert value.shape == (6, 4, 1)
    np.testing.assert_array_equal(
        np.array(value)[:, :, 0], np.array(torch_layout)[:, 0, :]
    )
