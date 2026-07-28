from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.nemotron_asr.checkpoint import (
    NemotronKeyError,
    convert_nemo_state_dict,
    expected_nemo_keys,
    map_nemo_key,
)


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
