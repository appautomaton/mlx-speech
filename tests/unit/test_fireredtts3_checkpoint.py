from __future__ import annotations

import io
import pickle
import sys
import types
import zipfile
from collections import OrderedDict

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.models.fireredtts3.checkpoint import (
    convert_component_arrays,
    load_torch_zip_state_dict,
)


class _FakeStorage:
    def __init__(self, key: str, values: np.ndarray) -> None:
        self.key = key
        self.values = values


class _FakeTensor:
    def __init__(self, storage: _FakeStorage, shape: tuple[int, ...]) -> None:
        self.storage = storage
        self.shape = shape


def _write_torch_zip(path, *, location: str = "cpu") -> None:
    torch_module = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch._utils")
    float_storage = type("FloatStorage", (), {"__module__": "torch"})

    def rebuild(storage, offset, size, stride, requires_grad, hooks):
        del offset, stride, requires_grad, hooks
        return _FakeTensor(storage, tuple(size))

    rebuild.__module__ = "torch._utils"
    rebuild.__name__ = "_rebuild_tensor_v2"
    rebuild.__qualname__ = "_rebuild_tensor_v2"
    torch_module.FloatStorage = float_storage
    torch_utils._rebuild_tensor_v2 = rebuild
    prior_torch = sys.modules.get("torch")
    prior_utils = sys.modules.get("torch._utils")
    sys.modules["torch"] = torch_module
    sys.modules["torch._utils"] = torch_utils
    try:
        storage = _FakeStorage("0", np.arange(6, dtype="<f4"))
        state = OrderedDict(weight=_FakeTensor(storage, (2, 3)))
        buffer = io.BytesIO()

        class _Pickler(pickle.Pickler):
            def persistent_id(self, obj):
                if isinstance(obj, _FakeStorage):
                    return (
                        "storage",
                        float_storage,
                        obj.key,
                        location,
                        obj.values.size,
                    )
                return None

            def reducer_override(self, obj):
                if isinstance(obj, _FakeTensor):
                    return (
                        rebuild,
                        (obj.storage, 0, obj.shape, (3, 1), False, OrderedDict()),
                    )
                return NotImplemented

        _Pickler(buffer, protocol=2).dump(state)
    finally:
        if prior_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = prior_torch
        if prior_utils is None:
            sys.modules.pop("torch._utils", None)
        else:
            sys.modules["torch._utils"] = prior_utils

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("embedding_model/data.pkl", buffer.getvalue())
        archive.writestr("embedding_model/data/0", storage.values.tobytes())


def test_restricted_torch_zip_reader_loads_contiguous_cpu_tensor(tmp_path) -> None:
    path = tmp_path / "speaker.bin"
    _write_torch_zip(path)
    state = load_torch_zip_state_dict(path)
    np.testing.assert_array_equal(state["weight"], np.arange(6).reshape(2, 3))
    assert state["weight"].dtype == np.float32


def test_restricted_torch_zip_reader_accepts_cuda_storage_metadata(tmp_path) -> None:
    path = tmp_path / "speaker.bin"
    _write_torch_zip(path, location="cuda:0")
    state = load_torch_zip_state_dict(path)
    np.testing.assert_array_equal(state["weight"], np.arange(6).reshape(2, 3))


def test_component_conversion_applies_mlx_layout_and_dtype_policy() -> None:
    core = convert_component_arrays(
        "core",
        {
            "dit.blocks.0.conv.block.0.weight": mx.ones((2, 3, 5)),
            "backbone_llm.layers.0.weight": mx.ones((2, 3)),
        },
    )
    assert core["dit.blocks.0.conv.block.0.weight"].shape == (2, 5, 3)
    assert core["backbone_llm.layers.0.weight"].dtype == mx.bfloat16

    speaker = convert_component_arrays(
        "speaker",
        {
            "head.conv1.weight": mx.ones((2, 3, 5, 7)),
            "head.bn1.running_mean": mx.ones((2,)),
            "head.bn1.num_batches_tracked": mx.array(3, dtype=mx.int64),
            "xvector.block1.tdnnd2.nonlinear1.batchnorm.weight": mx.ones(
                (4,)
            ),
            "xvector.transit1.linear.weight": mx.ones((2, 3, 1)),
        },
    )
    assert speaker["head.conv1.weight"].shape == (2, 5, 7, 3)
    assert speaker["head.conv1.weight"].dtype == mx.bfloat16
    assert speaker["head.bn1.running_mean"].dtype == mx.float32
    assert "head.bn1.num_batches_tracked" not in speaker
    assert "blocks.0.layers.1.nonlinear1.weight" in speaker
    assert speaker["transits.0.linear.weight"].shape == (2, 1, 3)


def test_component_conversion_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="unknown"):
        convert_component_arrays("unknown", {})
