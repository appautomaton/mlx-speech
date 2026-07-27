from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.nemotron_asr.config import ConformerArgs
from mlx_speech.models.nemotron_asr.encoder import (
    ConformerBlock,
    ConformerConvolution,
    FastConformerEncoder,
)


def _small_args(**overrides: object) -> ConformerArgs:
    values: dict[str, object] = {
        "feat_in": 16,
        "n_layers": 2,
        "d_model": 16,
        "n_heads": 4,
        "ff_expansion_factor": 2,
        "subsampling_conv_channels": 4,
        "att_context_size": ((8, 1),),
        "default_att_context_size": (8, 1),
        "pos_emb_max_len": 64,
    }
    values.update(overrides)
    return ConformerArgs(**values)


def test_default_config_matches_checkpoint() -> None:
    args = ConformerArgs()

    assert args.feat_in == 128
    assert args.n_layers == 24
    assert args.d_model == 1024
    assert args.n_heads == 8
    assert args.ff_expansion_factor == 4
    assert args.subsampling_factor == 8
    assert args.subsampling_conv_channels == 256
    assert args.conv_kernel_size == 9
    assert args.conv_context_size == "causal"
    assert args.conv_norm_type == "layer_norm"
    assert args.use_bias is False
    assert args.xscaling is False
    assert args.default_att_context_size == (56, 13)


def test_convolution_is_causal_layer_norm_with_checkpoint_name() -> None:
    convolution = ConformerConvolution(_small_args())

    assert convolution.kernel_size == 9
    assert convolution.pad_left == 8
    assert convolution.pad_right == 0
    assert convolution.depthwise_conv.groups == 16
    assert isinstance(convolution.batch_norm, nn.LayerNorm)
    for name in ("pointwise_conv1", "depthwise_conv", "pointwise_conv2"):
        assert "bias" not in getattr(convolution, name)


def test_convolution_does_not_observe_future_frames() -> None:
    rng = np.random.default_rng(21)
    values = rng.standard_normal((1, 19, 16)).astype(np.float32)
    changed = values.copy()
    changed[:, 12:] += 50.0
    convolution = ConformerConvolution(_small_args())

    first = convolution(mx.array(values))
    second = convolution(mx.array(changed))
    mx.eval(first, second)

    np.testing.assert_allclose(np.asarray(first)[:, :12], np.asarray(second)[:, :12])


def test_block_uses_macaron_order() -> None:
    calls: list[str] = []

    class Record(nn.Module):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def __call__(self, x: mx.array, *args: object) -> mx.array:
            calls.append(self.name)
            return x

    block = ConformerBlock(_small_args())
    block.norm_feed_forward1 = Record("norm_ff1")
    block.feed_forward1 = Record("ff1")
    block.norm_self_att = Record("norm_att")
    block.self_attn = Record("attn")
    block.norm_conv = Record("norm_conv")
    block.conv = Record("conv")
    block.norm_feed_forward2 = Record("norm_ff2")
    block.feed_forward2 = Record("ff2")
    block.norm_out = Record("norm_out")

    x = mx.zeros((1, 3, 16))
    block(x, mx.zeros((1, 5, 16)), mx.zeros((1, 1, 3, 3)))

    assert calls == [
        "norm_ff1",
        "ff1",
        "norm_att",
        "attn",
        "norm_conv",
        "conv",
        "norm_ff2",
        "ff2",
        "norm_out",
    ]


def test_block_parameter_paths_match_nemo() -> None:
    block = ConformerBlock(_small_args())
    names = {name for name, _ in tree_flatten(block.parameters())}

    expected = {
        "norm_feed_forward1.weight",
        "feed_forward1.linear1.weight",
        "feed_forward1.linear2.weight",
        "norm_self_att.weight",
        "self_attn.pos_bias_u",
        "self_attn.pos_bias_v",
        "self_attn.linear_q.weight",
        "self_attn.linear_k.weight",
        "self_attn.linear_v.weight",
        "self_attn.linear_out.weight",
        "self_attn.linear_pos.weight",
        "norm_conv.weight",
        "conv.pointwise_conv1.weight",
        "conv.depthwise_conv.weight",
        "conv.batch_norm.weight",
        "conv.pointwise_conv2.weight",
        "norm_feed_forward2.weight",
        "feed_forward2.linear1.weight",
        "feed_forward2.linear2.weight",
        "norm_out.weight",
    }
    assert expected <= names
    assert not any(name.endswith("linear1.bias") for name in names)
    projection_biases = {
        "conv.pointwise_conv1.bias",
        "conv.depthwise_conv.bias",
        "conv.pointwise_conv2.bias",
    }
    assert projection_biases.isdisjoint(names)


def test_encoder_forward_has_expected_shape_and_finite_values() -> None:
    encoder = FastConformerEncoder(_small_args())
    features = mx.linspace(-1.0, 1.0, 33 * 16).reshape(1, 33, 16)

    output, lengths = encoder(features, mx.array([33], dtype=mx.int32))
    mx.eval(output, lengths)

    assert output.shape == (1, 5, 16)
    assert lengths.tolist() == [5]
    assert mx.all(mx.isfinite(output)).item()


def test_encoder_trims_to_valid_length() -> None:
    encoder = FastConformerEncoder(_small_args(n_layers=1))
    features = mx.zeros((1, 33, 16))

    output, lengths = encoder(features, mx.array([17], dtype=mx.int32))
    mx.eval(output, lengths)

    assert output.shape == (1, 3, 16)
    assert lengths.tolist() == [3]


def test_encoder_rejects_deferred_batched_inference() -> None:
    encoder = FastConformerEncoder(_small_args(n_layers=0))

    with pytest.raises(ValueError, match="batch size 1"):
        encoder(mx.zeros((2, 17, 16)))
