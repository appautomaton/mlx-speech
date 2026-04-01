"""Minimal torch-free loader for PyTorch zip-archive state dicts."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Any
from zipfile import ZipFile

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class LoadedTorchArchiveStateDict:
    """Loaded tensor state from a PyTorch zip-archive checkpoint."""

    checkpoint_path: Path
    files: tuple[str, ...]
    weights: dict[str, mx.array]


@dataclass(frozen=True)
class _StorageRef:
    storage_name: str
    key: str
    location: str
    numel: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


_STORAGE_DTYPES: dict[str, np.dtype] = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "BFloat16Storage": np.dtype("<u2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
}


def _make_storage_type(name: str) -> type:
    return type(name, (), {"_torch_storage_name": name})


def _resolve_dtype(storage_type: type) -> np.dtype:
    storage_name = getattr(storage_type, "_torch_storage_name", storage_type.__name__)
    try:
        return _STORAGE_DTYPES[storage_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported PyTorch storage type: {storage_name}") from exc


def _rebuild_tensor_v2(
    storage: _StorageRef,
    storage_offset: int,
    size: tuple[int, ...] | list[int],
    stride: tuple[int, ...] | list[int],
    requires_grad: bool,
    backward_hooks: Any,
    metadata: Any = None,
) -> _TensorRef:
    _ = requires_grad, backward_hooks, metadata
    return _TensorRef(
        storage=storage,
        storage_offset=int(storage_offset),
        size=tuple(int(dim) for dim in size),
        stride=tuple(int(dim) for dim in stride),
    )


def _rebuild_parameter(
    data: _TensorRef,
    requires_grad: bool,
    backward_hooks: Any,
) -> _TensorRef:
    _ = requires_grad, backward_hooks
    return data


class _TorchArchiveUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name in {
            "_rebuild_tensor_v2",
            "_rebuild_tensor",
            "_rebuild_qtensor",
        }:
            return _rebuild_tensor_v2
        if module == "torch._utils" and name in {
            "_rebuild_parameter",
            "_rebuild_parameter_with_state",
        }:
            return _rebuild_parameter
        if module == "torch" and name.endswith("Storage"):
            return _make_storage_type(name)
        raise ValueError(f"Unsupported class in PyTorch archive pickle: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        if not isinstance(pid, tuple) or not pid:
            raise ValueError(f"Unsupported persistent id in PyTorch archive: {pid!r}")
        if pid[0] != "storage":
            raise ValueError(f"Unsupported persistent tuple in PyTorch archive: {pid!r}")
        _, storage_type, key, location, numel, *_ = pid
        return _StorageRef(
            storage_name=getattr(storage_type, "_torch_storage_name", storage_type.__name__),
            key=str(key),
            location=str(location),
            numel=int(numel),
        )


def _is_contiguous(size: tuple[int, ...], stride: tuple[int, ...]) -> bool:
    expected: list[int] = []
    running = 1
    for dim in reversed(size):
        expected.append(running)
        running *= max(int(dim), 1)
    return tuple(reversed(expected)) == tuple(stride)


def _materialize_tensor(zip_file: ZipFile, root_prefix: str, tensor_ref: _TensorRef) -> mx.array:
    data_path = f"{root_prefix}data/{tensor_ref.storage.key}"
    raw = zip_file.read(data_path)
    storage_dtype = _STORAGE_DTYPES[tensor_ref.storage.storage_name]
    storage = np.frombuffer(raw, dtype=storage_dtype, count=tensor_ref.storage.numel)

    if tensor_ref.storage.storage_name == "BFloat16Storage":
        # Preserve exact BF16 bits before mlx conversion.
        view = storage[tensor_ref.storage_offset :]
        shaped = np.lib.stride_tricks.as_strided(
            view,
            shape=tensor_ref.size,
            strides=tuple(step * storage_dtype.itemsize for step in tensor_ref.stride),
        ).copy()
        float32 = (shaped.astype(np.uint32) << 16).view(np.float32)
        return mx.array(float32)

    if _is_contiguous(tensor_ref.size, tensor_ref.stride):
        count = int(np.prod(tensor_ref.size, dtype=np.int64)) if tensor_ref.size else 1
        array = storage[tensor_ref.storage_offset : tensor_ref.storage_offset + count]
        shaped = array.reshape(tensor_ref.size)
    else:
        view = storage[tensor_ref.storage_offset :]
        shaped = np.lib.stride_tricks.as_strided(
            view,
            shape=tensor_ref.size,
            strides=tuple(step * storage_dtype.itemsize for step in tensor_ref.stride),
        ).copy()
    return mx.array(shaped)


def load_torch_archive_state_dict(checkpoint_path: str | Path) -> LoadedTorchArchiveStateDict:
    """Load tensors from a PyTorch zip-archive checkpoint without torch."""

    resolved = Path(checkpoint_path)
    if not resolved.exists():
        raise FileNotFoundError(f"PyTorch checkpoint not found: {resolved}")

    with ZipFile(resolved) as zip_file:
        names = tuple(zip_file.namelist())
        data_pickle = next((name for name in names if name.endswith("data.pkl")), None)
        if data_pickle is None:
            raise ValueError(f"{resolved} does not look like a PyTorch zip checkpoint.")
        root_prefix = data_pickle.removesuffix("data.pkl")
        state = _TorchArchiveUnpickler(zip_file.open(data_pickle)).load()

        if not isinstance(state, (OrderedDict, dict)):
            raise ValueError(
                f"Expected dict-like state_dict in {resolved}, got {type(state).__name__}."
            )

        weights: dict[str, mx.array] = {}
        for key, value in state.items():
            if not isinstance(value, _TensorRef):
                raise ValueError(
                    f"Unsupported value type for key {key!r} in {resolved}: "
                    f"{type(value).__name__}."
                )
            weights[str(key)] = _materialize_tensor(zip_file, root_prefix, value)

    return LoadedTorchArchiveStateDict(
        checkpoint_path=resolved,
        files=names,
        weights=weights,
    )


__all__ = [
    "LoadedTorchArchiveStateDict",
    "load_torch_archive_state_dict",
]
