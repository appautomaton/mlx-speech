from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_speech.models.granite_speech_asr.config import (
    GraniteSpeechConfig,
    GraniteSpeechEncoderConfig,
    GraniteSpeechProjectorConfig,
    GraniteSpeechTextConfig,
)
from mlx_speech.models.granite_speech_asr.projector import GraniteSpeechProjector


def _tiny_config() -> GraniteSpeechConfig:
    return GraniteSpeechConfig(
        encoder=GraniteSpeechEncoderConfig(hidden_dim=6, output_dim=3),
        projector=GraniteSpeechProjectorConfig(
            hidden_size=8,
            encoder_hidden_size=6,
            intermediate_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            layer_norm_eps=1e-6,
        ),
        text=GraniteSpeechTextConfig(hidden_size=10),
        window_size=15,
        downsample_rate=5,
    )


def test_granite_projector_pads_windows_and_returns_text_hidden_size():
    projector = GraniteSpeechProjector(_tiny_config())
    encoder_states = mx.zeros((1, 16, 6), dtype=mx.float32)

    projected = projector(encoder_states)
    mx.eval(projected)

    assert projector.query.shape == (1, 3, 8)
    assert projected.shape == (1, 6, 10)
    assert mx.all(mx.isfinite(projected)).item()


def test_granite_projector_exact_window_returns_three_queries():
    projector = GraniteSpeechProjector(_tiny_config())
    encoder_states = mx.zeros((2, 15, 6), dtype=mx.float32)

    projected = projector(encoder_states)
    mx.eval(projected)

    assert projected.shape == (2, 3, 10)


def test_granite_projector_parameter_names_match_checkpoint_subtree():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.projector = GraniteSpeechProjector(_tiny_config())

    params = tree_flatten(Wrapper().parameters(), destination={})
    expected = {
        "projector.query",
        "projector.qformer.layernorm.weight",
        "projector.qformer.encoder.layer.0.attention.attention.query.weight",
        "projector.qformer.encoder.layer.0.crossattention.attention.key.weight",
        "projector.qformer.encoder.layer.0.intermediate_query.dense.bias",
        "projector.qformer.encoder.layer.1.output_query.LayerNorm.weight",
        "projector.linear.bias",
    }

    assert expected <= set(params)
