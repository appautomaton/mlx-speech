from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx_speech.models.nemotron_asr.checkpoint import (
    NemotronKeyError,
    QuantizationConfig,
    convert_nemo_state_dict,
    expected_nemo_keys,
    get_quantization_config,
    load_state_dict_strict,
    map_nemo_key,
    quantize_nemotron_model,
)
from mlx_speech.models.nemotron_asr.config import NemotronASRConfig


class _TinyQuantizableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(64, 4)
        self.embedding = nn.Embedding(16, 64)
        self.unaligned = nn.Linear(6, 4)


def test_expected_schema_accounts_for_every_source_tensor() -> None:
    keys = expected_nemo_keys()

    assert len(keys) == 657
    assert all(map_nemo_key(key).source == key for key in keys)


def test_unknown_or_missing_keys_raise() -> None:
    with pytest.raises(NemotronKeyError, match="unmapped"):
        map_nemo_key("encoder.layers.0.not_a_real_parameter")
    with pytest.raises(NemotronKeyError, match="schema mismatch"):
        convert_nemo_state_dict({"unknown.weight": np.zeros((1,), np.float32)})


def test_conv_layout_transform_is_explicit() -> None:
    mapping = map_nemo_key("encoder.layers.0.conv.depthwise_conv.weight")

    assert mapping.destination == mapping.source
    assert mapping.transform == "conv_layout"


def test_small_complete_schema_converts_and_merges_lstm_biases() -> None:
    keys = expected_nemo_keys(n_layers=0, rnn_layers=1)
    weights: dict[str, np.ndarray] = {}
    for key in keys:
        if key.endswith("conv.0.weight"):
            weights[key] = np.zeros((2, 1, 3, 3), np.float32)
        elif ".conv." in key and key.endswith(".weight"):
            weights[key] = np.zeros((2, 1, 3), np.float32)
        else:
            weights[key] = np.ones((2,), np.float32)

    converted, report = convert_nemo_state_dict(
        weights, dtype=mx.float32, n_layers=0, rnn_layers=1
    )
    mx.eval(converted)

    assert report.source_count == len(keys)
    assert report.destination_count == len(keys) - 1
    bias = converted["decoder.prediction.dec_rnn.lstm.0.bias"]
    np.testing.assert_array_equal(np.asarray(bias), np.full((2,), 2.0))
    assert converted["encoder.pre_encode.conv.0.weight"].shape == (2, 3, 3, 1)


def test_quantization_config_round_trips_through_model_config() -> None:
    quantization = QuantizationConfig(bits=8, group_size=64, mode="affine")
    config = NemotronASRConfig(quantization=quantization.to_dict())

    restored = NemotronASRConfig.from_dict(config.to_dict())

    assert get_quantization_config(restored) == quantization
    assert "quantization" not in NemotronASRConfig().to_dict()


def test_quantized_saved_layout_reconstructs_and_loads_strictly() -> None:
    quantization = QuantizationConfig(bits=8, group_size=32, mode="affine")
    original = _TinyQuantizableModel()
    quantize_nemotron_model(original, quantization)
    state = tree_flatten(original.parameters(), destination={})

    restored = _TinyQuantizableModel()
    quantize_nemotron_model(restored, quantization, state_dict=state)
    report = load_state_dict_strict(restored, state)

    assert report.is_exact_match
    assert isinstance(restored.linear, nn.QuantizedLinear)
    assert isinstance(restored.embedding, nn.QuantizedEmbedding)
    assert isinstance(restored.unaligned, nn.Linear)
    assert {"linear.scales", "linear.biases"} <= set(state)

    inputs = mx.arange(128, dtype=mx.float32).reshape(2, 64)
    expected = original.linear(inputs)
    actual = restored.linear(inputs)
    mx.eval(expected, actual)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
