"""Fish codec decoder kernels: each faster form computes what the plain one does."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.fish_s2_pro.codec_model import (
    CausalWNConv1d,
    ConvTranspose1dTorch,
    _conv_transpose1d_torch_to_mlx,
    _normalize_weight,
    snake,
)


@pytest.fixture(autouse=True)
def on_cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("stride", [2, 4, 8])
def test_stride_phase_upsampling_equals_conv_transpose(stride):
    mx.random.seed(stride)
    conv = ConvTranspose1dTorch(12, 5, kernel_size=2 * stride, stride=stride)
    conv.bias = mx.random.normal((5,))
    x = mx.random.normal((2, 12, 9))  # [batch, channels, steps]

    reference = mx.conv_transpose1d(
        x.swapaxes(1, 2), _conv_transpose1d_torch_to_mlx(conv.weight), stride=stride
    ) + conv.bias

    assert conv._stride_phases()
    np.testing.assert_allclose(
        np.array(conv(x).swapaxes(1, 2)), np.array(reference), rtol=1e-5, atol=1e-5
    )


def test_other_transposed_convolutions_keep_the_general_kernel():
    conv = ConvTranspose1dTorch(4, 4, kernel_size=3, stride=2)
    assert not conv._stride_phases()
    assert conv(mx.random.normal((1, 4, 5))).shape == (1, 4, 11)


def test_weight_norm_is_folded_once_and_again_for_new_weights():
    conv = CausalWNConv1d(6, 6, kernel_size=3)
    x = mx.random.normal((1, 6, 10))

    conv(x)
    first = conv.conv.conv.weight
    conv(x)
    assert conv.conv.conv.weight is first  # reused, not recomputed

    conv.weight_v = mx.random.normal(conv.weight_v.shape)
    conv(x)
    expected = conv.weight_g * conv.weight_v / _normalize_weight(conv.weight_v)
    assert conv.conv.conv.weight is not first
    np.testing.assert_allclose(
        np.array(conv.conv.conv.weight), np.array(expected), rtol=1e-6
    )
    assert "_folded" not in dict(conv.parameters())


def test_fused_snake_matches_its_formula():
    x = mx.random.normal((2, 7, 50)) * 3
    alpha = mx.random.uniform(0.2, 2.0, (1, 7, 1))

    plain = x + mx.reciprocal(alpha + 1e-9) * mx.power(mx.sin(alpha * x), 2)

    np.testing.assert_allclose(np.array(snake(x, alpha)), np.array(plain), atol=1e-5)
