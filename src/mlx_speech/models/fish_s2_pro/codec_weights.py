from __future__ import annotations

import io
import json
import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .codec_config import FishCodecConfig


class FloatStorage:
    pass


class BFloat16Storage:
    pass


class BoolStorage:
    pass


class HalfStorage:
    pass


class LongStorage:
    pass


class IntStorage:
    pass


def _cleanup_temp_file(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def save_codec_assets(
    output_dir: str | Path,
    weights: dict[str, mx.array],
    config: FishCodecConfig,
) -> None:
    resolved = Path(output_dir)
    if not weights:
        raise ValueError("No codec weights to save")

    resolved.mkdir(parents=True, exist_ok=True)
    config_path = resolved / "config.json"
    weights_path = resolved / "model.safetensors"
    temp_config_path = resolved / ".config.json.tmp"
    temp_weights_path = resolved / ".model.tmp.safetensors"

    try:
        temp_config_path.write_text(
            json.dumps(config.to_dict(), indent=2),
            encoding="utf-8",
        )
        mx.save_safetensors(str(temp_weights_path), weights)
        temp_weights_path.replace(weights_path)
        temp_config_path.replace(config_path)
    except Exception:
        _cleanup_temp_file(temp_config_path)
        _cleanup_temp_file(temp_weights_path)
        raise


def _storage_dtype(storage_type: object) -> np.dtype:
    name = getattr(storage_type, "__name__", storage_type)
    dtype_map = {
        "HalfStorage": np.float16,
        "FloatStorage": np.float32,
        "BFloat16Storage": np.uint16,
        "BoolStorage": np.bool_,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
    }
    try:
        return dtype_map[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported Fish codec storage type: {name}") from exc


def _bfloat16_to_float32(values: np.ndarray) -> np.ndarray:
    widened = values.astype(np.uint32) << 16
    return widened.view(np.float32)


def _rebuild_tensor_v2(
    storage: np.ndarray,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: object,
) -> np.ndarray:
    del requires_grad, backward_hooks
    base = np.asarray(storage)
    if base.dtype == np.uint16:
        base = _bfloat16_to_float32(base)
    view = base[storage_offset:]
    rebuilt = np.lib.stride_tricks.as_strided(
        view,
        shape=tuple(size),
        strides=tuple(step * view.dtype.itemsize for step in stride),
    )
    return np.array(rebuilt, copy=True)


def convert_codec_pth_to_assets(codec_pth: str | Path, output_dir: str | Path) -> None:
    resolved = Path(codec_pth)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing Fish codec checkpoint: {resolved}")

    with zipfile.ZipFile(resolved) as zf:
        try:
            root = next(
                name.split("/")[0]
                for name in zf.namelist()
                if name.endswith("data.pkl")
            )
        except StopIteration as exc:
            raise ValueError(
                f"Missing data.pkl in Fish codec checkpoint: {resolved}"
            ) from exc

        raw = zf.read(f"{root}/data.pkl")
        storages: dict[str, np.ndarray] = {}

        class _ArchiveUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "collections" and name == "OrderedDict":
                    return OrderedDict
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor_v2
                if module == "torch":
                    storage_types = {
                        "FloatStorage": FloatStorage,
                        "BFloat16Storage": BFloat16Storage,
                        "BoolStorage": BoolStorage,
                        "HalfStorage": HalfStorage,
                        "LongStorage": LongStorage,
                        "IntStorage": IntStorage,
                    }
                    if name in storage_types:
                        return storage_types[name]
                raise ModuleNotFoundError(f"Unsupported pickle global: {module}.{name}")

            def persistent_load(self, pid):
                kind, storage_type, storage_key, location, numel = pid
                del kind, location
                dtype = _storage_dtype(storage_type)
                if storage_key not in storages:
                    with zf.open(f"{root}/data/{storage_key}") as handle:
                        storages[storage_key] = np.frombuffer(
                            handle.read(), dtype=dtype, count=numel
                        )
                return storages[storage_key]

        state = _ArchiveUnpickler(io.BytesIO(raw)).load()
        weights = {key: mx.array(np.array(tensor)) for key, tensor in state.items()}
        save_codec_assets(output_dir, weights, FishCodecConfig())
