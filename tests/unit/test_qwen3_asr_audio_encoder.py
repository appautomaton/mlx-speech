from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
import numpy as np
import pytest

from mlx_speech.models.qwen3_asr.audio_encoder import (
    Qwen3ASRAudioAttention,
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


@pytest.mark.parametrize("lengths", [[7, 40, 100, 1817], [7, 19, 41]])
def test_qwen3_asr_audio_encoder_batch_matches_per_clip(lengths, monkeypatch):
    # CPU is the float32 oracle; Metal GEMMs may use batch-dependent TF32.
    with mx.stream(mx.cpu):
        mx.random.seed(71)
        encoder = Qwen3ASRAudioEncoder(_tiny_config(output_dim=16))
        # Nonzero biases expose artificial right-boundary activations after padding.
        for conv in (encoder.conv2d1, encoder.conv2d2, encoder.conv2d3):
            conv.bias = mx.full(conv.bias.shape, 0.1)
        features = [mx.random.normal((8, length)) for length in lengths]
        expected = [encoder(clip).last_hidden_state for clip in features]
        layer_call = Qwen3ASRAudioEncoderLayer.__call__
        calls = []

        def checked(layer, hidden_states, **kwargs):
            calls.append(hidden_states.shape)
            return layer_call(layer, hidden_states, **kwargs)

        monkeypatch.setattr(Qwen3ASRAudioEncoderLayer, "__call__", checked)
        actual = encoder.encode_batch(features)
        mx.eval(expected, actual)

        assert len(calls) == len(encoder.layers)
        assert all(len(shape) == 3 for shape in calls)
        assert len(actual) == len(expected)
        for batch_row, single_row in zip(actual, expected, strict=True):
            assert batch_row.shape == single_row.shape
            np.testing.assert_allclose(np.array(batch_row), np.array(single_row), atol=1e-5, rtol=1e-5)


def test_qwen3_asr_audio_encoder_batch_uses_weight_dtype():
    mx.random.seed(72)
    encoder = Qwen3ASRAudioEncoder(_tiny_config(output_dim=16))
    encoder.set_dtype(mx.bfloat16)
    outputs = encoder.encode_batch([mx.random.normal((8, length)) for length in [7, 100, 817]])
    mx.eval(outputs)

    assert [row.shape for row in outputs] == [(1, 16), (13, 16), (107, 16)]
    assert all(row.dtype == mx.bfloat16 for row in outputs)
    assert all(mx.all(mx.isfinite(row)).item() for row in outputs)
    # The legacy single-clip API still follows the caller's input dtype.
    assert encoder(mx.zeros((8, 40), dtype=mx.float32)).last_hidden_state.dtype == mx.float32


@pytest.mark.parametrize("length", [1, 7, 104])
def test_qwen3_asr_audio_attention_sdpa_matches_eager(length):
    mx.random.seed(73)
    attention = Qwen3ASRAudioAttention(_tiny_config())
    x = mx.random.normal((length, attention.embed_dim))
    q, k, v = [
        projection(x).reshape(length, attention.num_heads, attention.head_dim).transpose(1, 0, 2)
        for projection in (attention.q_proj, attention.k_proj, attention.v_proj)
    ]
    scores = (q @ k.transpose(0, 2, 1)) * attention.scale
    weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    eager = attention.out_proj((weights @ v).transpose(1, 0, 2).reshape(length, attention.embed_dim))
    actual = attention._attend_block(x)
    mx.eval(eager, actual)

    np.testing.assert_allclose(np.array(actual), np.array(eager), atol=1e-5, rtol=1e-5)


def test_qwen3_asr_audio_encoder_batch_validates_features():
    encoder = Qwen3ASRAudioEncoder(_tiny_config(output_dim=16))
    assert encoder.encode_batch([]) == []
    with pytest.raises(ValueError, match=r"\[mel, frames\]"):
        encoder.encode_batch([mx.zeros((1, 8, 40))])
    with pytest.raises(ValueError, match="mel bins"):
        encoder.encode_batch([mx.zeros((7, 40))])
    with pytest.raises(ValueError, match="positive"):
        encoder.encode_batch([mx.zeros((8, 0))])
