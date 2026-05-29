from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from mlx_speech.models.granite_speech_asr.config import GraniteSpeechEncoderConfig
from mlx_speech.models.granite_speech_asr.encoder import GraniteSpeechEncoder


def _tiny_config(**overrides) -> GraniteSpeechEncoderConfig:
    values = {
        "input_dim": 8,
        "hidden_dim": 16,
        "output_dim": 5,
        "num_layers": 2,
        "num_heads": 2,
        "dim_head": 4,
        "feedforward_mult": 2,
        "conv_expansion_factor": 2,
        "conv_kernel_size": 3,
        "context_size": 4,
        "max_pos_emb": 8,
        "dropout": 0.0,
    }
    values.update(overrides)
    return GraniteSpeechEncoderConfig(**values)


def test_granite_encoder_forward_preserves_hidden_shape():
    encoder = GraniteSpeechEncoder(_tiny_config())
    x = mx.zeros((2, 8, 8), dtype=mx.float32)

    y = encoder(x)
    mx.eval(y)

    assert y.shape == (2, 8, 16)
    assert mx.all(mx.isfinite(y)).item()


def test_granite_encoder_handles_non_multiple_context_lengths():
    encoder = GraniteSpeechEncoder(_tiny_config(context_size=4))
    x = mx.zeros((1, 5, 8), dtype=mx.float32)

    y = encoder(x)
    mx.eval(y)

    assert y.shape == (1, 5, 16)


def test_granite_encoder_parameter_names_match_checkpoint_subtree():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GraniteSpeechEncoder(_tiny_config())

    params = tree_flatten(Wrapper().parameters(), destination={})

    expected = {
        "encoder.input_linear.weight",
        "encoder.layers.0.attn.pre_norm.weight",
        "encoder.layers.0.attn.to_q.weight",
        "encoder.layers.0.attn.to_kv.weight",
        "encoder.layers.0.attn.to_out.bias",
        "encoder.layers.0.attn.rel_pos_emb.weight",
        "encoder.layers.0.conv.up_conv.weight",
        "encoder.layers.0.conv.depth_conv.conv.weight",
        "encoder.layers.0.conv.batch_norm.running_mean",
        "encoder.layers.0.conv.down_conv.bias",
        "encoder.layers.0.ff1.up_proj.weight",
        "encoder.layers.0.ff2.down_proj.bias",
        "encoder.layers.0.post_norm.weight",
        "encoder.out.weight",
        "encoder.out_mid.bias",
    }

    assert expected <= set(params)


def test_granite_encoder_rejects_non_sequence_input():
    encoder = GraniteSpeechEncoder(_tiny_config())

    with pytest.raises(ValueError, match=r"\[B, T, C\]"):
        encoder(mx.zeros((8, 8), dtype=mx.float32))
