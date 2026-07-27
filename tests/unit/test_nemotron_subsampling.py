from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.nemotron_asr.subsampling import (
    CausalDwStridingSubsampling,
    subsampled_length,
)

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "nemotron" / "subsampling.npz"


def _load_reference_module(fixture: dict[str, np.ndarray]) -> CausalDwStridingSubsampling:
    module = CausalDwStridingSubsampling(
        feat_in=int(fixture["feat_in"]),
        d_model=int(fixture["d_model"]),
        conv_channels=int(fixture["conv_channels"]),
    )
    for index in (0, 2, 3, 5, 6):
        layer = module.conv[index]
        torch_weight = fixture[f"conv_{index}_weight"]
        layer.weight = mx.array(np.transpose(torch_weight, (0, 2, 3, 1)))
        layer.bias = mx.array(fixture[f"conv_{index}_bias"])
    module.out.weight = mx.array(fixture["out_weight"])
    module.out.bias = mx.array(fixture["out_bias"])
    return module


@pytest.mark.parametrize("length", [0, 1, 2, 3, 7, 8, 9, 25, 100, 101])
def test_output_length_matches_nemo_recurrence(length: int) -> None:
    expected = length
    for _ in range(3):
        expected = (expected + 2 + 1 - 3) // 2 + 1
    assert subsampled_length(length) == expected


def test_architecture_matches_dw_striding_layout() -> None:
    module = CausalDwStridingSubsampling()

    assert module._strided_indices == frozenset({0, 2, 5})
    assert module.conv[0].weight.shape == (256, 3, 3, 1)
    assert module.conv[2].weight.shape == (256, 3, 3, 1)
    assert module.conv[2].groups == 256
    assert module.conv[3].weight.shape == (256, 1, 1, 256)
    assert module.out.weight.shape == (1024, 256 * 17)


def test_forward_matches_captured_torch_reference() -> None:
    with np.load(_FIXTURE) as data:
        fixture = {key: data[key] for key in data.files}

    module = _load_reference_module(fixture)
    output, lengths = module(mx.array(fixture["features"]), mx.array(fixture["lengths"]))
    mx.eval(output, lengths)

    np.testing.assert_allclose(
        np.asarray(output), fixture["output"], rtol=2e-5, atol=2e-5
    )
    np.testing.assert_array_equal(np.asarray(lengths), fixture["output_lengths"])


def test_channel_major_flatten_order() -> None:
    module = CausalDwStridingSubsampling(feat_in=11, d_model=5, conv_channels=2)
    batch, time, frequency, channels = 1, 3, 4, 2
    x = mx.arange(batch * time * frequency * channels).reshape(
        batch, time, frequency, channels
    )

    flattened = mx.transpose(x, (0, 1, 3, 2)).reshape(
        batch, time, channels * frequency
    )
    mx.eval(flattened)

    expected = np.asarray(x).transpose(0, 1, 3, 2).reshape(batch, time, -1)
    np.testing.assert_array_equal(np.asarray(flattened), expected)
    assert module.out.weight.shape[-1] == 2 * subsampled_length(11)


def test_future_frames_do_not_change_earlier_valid_outputs() -> None:
    rng = np.random.default_rng(7)
    features = rng.standard_normal((1, 33, 11)).astype(np.float32)
    changed = features.copy()
    changed[:, 25:] += 100.0
    module = CausalDwStridingSubsampling(feat_in=11, d_model=5, conv_channels=2)

    first, _ = module(mx.array(features), mx.array([33], dtype=mx.int32))
    second, _ = module(mx.array(changed), mx.array([33], dtype=mx.int32))
    mx.eval(first, second)

    # Encoder frames whose receptive fields end before mel frame 25 are unchanged.
    np.testing.assert_allclose(np.asarray(first)[:, :4], np.asarray(second)[:, :4])


def test_rejects_wrong_feature_shape() -> None:
    module = CausalDwStridingSubsampling(feat_in=11, d_model=5, conv_channels=2)
    with pytest.raises(ValueError, match="mel bins"):
        module(mx.zeros((1, 8, 10)), mx.array([8], dtype=mx.int32))
