from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
import pytest

from mlx_speech.models.qwen3_asr.audio_encoder import (
    Qwen3ASRAudioEncoder,
    Qwen3ASRAudioEncoderLayer,
    Qwen3ASRAudioEncoderOutput,
    _attention_block_lengths,
    _conv2d_subsampled_length,
)
from mlx_speech.models.qwen3_asr.config import Qwen3ASRAudioConfig


def _tiny_config(**overrides) -> Qwen3ASRAudioConfig:
    values = {
        "d_model": 16,
        "num_mel_bins": 8,
        "encoder_layers": 2,
        "encoder_attention_heads": 4,
        "encoder_ffn_dim": 32,
        "downsample_hidden_size": 4,
        "output_dim": 2048,
        "max_source_positions": 64,
        "n_window": 50,
        "n_window_infer": 800,
        "conv_chunksize": 2,
    }
    values.update(overrides)
    return Qwen3ASRAudioConfig(**values)


def test_qwen3_asr_audio_encoder_builds_reference_structure_from_config():
    encoder = Qwen3ASRAudioEncoder(_tiny_config(encoder_layers=24))

    assert len(encoder.layers) == 24
    assert all(isinstance(layer, Qwen3ASRAudioEncoderLayer) for layer in encoder.layers)
    assert encoder.conv2d1.stride == (2, 2)
    assert encoder.conv2d2.stride == (2, 2)
    assert encoder.conv2d3.stride == (2, 2)
    assert encoder.conv_out.weight.shape == (16, 4 * _conv2d_subsampled_length(8))
    assert encoder.proj2.weight.shape == (2048, 16)


def test_qwen3_asr_audio_encoder_shape_forward_outputs_2048_width():
    encoder = Qwen3ASRAudioEncoder(_tiny_config())
    features = mx.zeros((8, 40), dtype=mx.float32)

    output = encoder(features, feature_lens=40)
    mx.eval(output.last_hidden_state)

    assert isinstance(output, Qwen3ASRAudioEncoderOutput)
    assert output.last_hidden_state.shape == (5, 2048)
    assert mx.all(mx.isfinite(output.last_hidden_state)).item()


def test_qwen3_asr_audio_encoder_uses_feature_length_and_masks_tail():
    encoder = Qwen3ASRAudioEncoder(_tiny_config())
    features = mx.zeros((1, 8, 64), dtype=mx.float32)

    output = encoder(features, feature_lens=[20])
    mx.eval(output.last_hidden_state)

    assert output.last_hidden_state.shape == (3, 2048)


def test_qwen3_asr_audio_encoder_output_length_formula():
    encoder = Qwen3ASRAudioEncoder(_tiny_config())

    assert encoder.output_lengths(40) == 5
    assert encoder.output_lengths(100) == 13


def test_qwen3_asr_audio_encoder_parameter_names_match_checkpoint_surface():
    encoder = Qwen3ASRAudioEncoder(_tiny_config())
    params = tree_flatten(encoder.parameters(), destination={})

    expected = {
        "conv2d1.weight",
        "conv2d2.weight",
        "conv2d3.weight",
        "conv_out.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.bias",
        "layers.0.self_attn_layer_norm.weight",
        "layers.0.fc1.weight",
        "layers.0.fc2.bias",
        "ln_post.weight",
        "proj1.weight",
        "proj2.bias",
    }

    assert expected <= set(params)


def test_qwen3_asr_audio_encoder_rejects_invalid_shapes_and_batching():
    encoder = Qwen3ASRAudioEncoder(_tiny_config())

    with pytest.raises(ValueError, match=r"\[mel, frames\]"):
        encoder(mx.zeros((8,), dtype=mx.float32))

    with pytest.raises(ValueError, match="one audio input"):
        encoder(mx.zeros((2, 8, 40), dtype=mx.float32))

    with pytest.raises(ValueError, match="mel bins"):
        encoder(mx.zeros((7, 40), dtype=mx.float32))

    with pytest.raises(ValueError, match="positive"):
        encoder(mx.zeros((8, 40), dtype=mx.float32), feature_lens=0)


def test_qwen3_asr_audio_attention_block_lengths_group_infer_windows():
    assert _attention_block_lengths(
        total_length=260,
        max_chunk_aftercnn=13,
        n_window=50,
        n_window_infer=800,
    ) == [104, 104, 52]
