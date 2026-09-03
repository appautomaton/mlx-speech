"""Strict checkpoint conversion for FireRedTTS3 Base."""

from __future__ import annotations

import io
import json
import math
import pickle
import shutil
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import (
    ARTIFACT_FILES,
    FORMAT_VERSION,
    MODEL_TYPE,
    TOKENIZER_FILES,
    FireRedTTS3Config,
)


SOURCE_REVISION = "dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610"
SOURCE_CODE_REVISION = "1d32ba780da6af37a71bdfd9c68c12003e908a46"
EXPECTED_SOURCE_TENSORS = {"core": 677, "redae": 458, "speaker": 937}

_SOURCE_FILES = {
    "core": Path("fireredtts3_base/model.safetensors"),
    "core_config": Path("fireredtts3_base/config.json"),
    "redae": Path("redae/model.safetensors"),
    "redae_config": Path("redae/config.json"),
    "speaker": Path("campp/campplus_voxceleb.bin"),
}

_QWEN_CONFIG = {
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1_000_000,
    "rms_norm_eps": 1e-6,
}

_SPEAKER_CONFIG = {
    "feature_dim": 80,
    "embedding_size": 512,
    "sample_rate": 16000,
    "growth_rate": 32,
    "init_channels": 128,
    "block_layers": [12, 24, 16],
    "block_dilations": [1, 2, 2],
}


@dataclass(frozen=True)
class _StorageRef:
    key: str
    dtype: np.dtype[Any]
    size: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]


class _FloatStorage:
    dtype = np.dtype("<f4")


class _LongStorage:
    dtype = np.dtype("<i8")


def _rebuild_tensor_v2(
    storage: _StorageRef,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: Any,
) -> _TensorRef:
    del requires_grad, backward_hooks
    return _TensorRef(storage, int(storage_offset), tuple(size), tuple(stride))


class _TorchStateUnpickler(pickle.Unpickler):
    """Decode the restricted tensor vocabulary used by the official CAM++ file."""

    _ALLOWED_GLOBALS = {
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        ("torch", "FloatStorage"): _FloatStorage,
        ("torch", "LongStorage"): _LongStorage,
        ("collections", "OrderedDict"): OrderedDict,
    }

    def find_class(self, module: str, name: str) -> Any:
        try:
            return self._ALLOWED_GLOBALS[(module, name)]
        except KeyError as error:
            raise pickle.UnpicklingError(
                f"unsupported global in CAM++ checkpoint: {module}.{name}"
            ) from error

    def persistent_load(self, persistent_id: Any) -> _StorageRef:
        if not isinstance(persistent_id, tuple) or len(persistent_id) != 5:
            raise pickle.UnpicklingError("invalid CAM++ storage reference")
        tag, storage_type, key, location, size = persistent_id
        valid_location = location == "cpu" or (
            isinstance(location, str)
            and location.startswith("cuda:")
            and location.removeprefix("cuda:").isdigit()
        )
        if tag != "storage" or not valid_location:
            raise pickle.UnpicklingError(
                f"unsupported CAM++ storage descriptor: {persistent_id!r}"
            )
        dtype = getattr(storage_type, "dtype", None)
        if dtype not in {_FloatStorage.dtype, _LongStorage.dtype}:
            raise pickle.UnpicklingError("unsupported CAM++ storage dtype")
        return _StorageRef(str(key), dtype, int(size))


def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    size = 1
    for dimension in reversed(shape):
        stride.append(size)
        size *= dimension
    return tuple(reversed(stride))


def load_torch_zip_state_dict(path: str | Path) -> dict[str, np.ndarray[Any, Any]]:
    """Load the constrained CPU tensor state dict used by CAM++ without Torch."""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"CAM++ checkpoint not found: {checkpoint_path}")
    try:
        archive = zipfile.ZipFile(checkpoint_path)
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"CAM++ checkpoint is not a PyTorch zip archive: {path}"
        ) from error
    with archive:
        pickle_members = [
            name for name in archive.namelist() if name.endswith("/data.pkl")
        ]
        if len(pickle_members) != 1:
            raise ValueError("CAM++ checkpoint must contain exactly one data.pkl")
        prefix = pickle_members[0].removesuffix("data.pkl")
        unpickler = _TorchStateUnpickler(io.BytesIO(archive.read(pickle_members[0])))
        references = unpickler.load()
        if not isinstance(references, dict):
            raise ValueError("CAM++ checkpoint root must be a state dict")

        arrays: dict[str, np.ndarray[Any, Any]] = {}
        for name, tensor in references.items():
            if not isinstance(name, str) or not isinstance(tensor, _TensorRef):
                raise ValueError("CAM++ state dict contains a non-tensor entry")
            if tensor.stride != _contiguous_stride(tensor.shape):
                raise ValueError(f"CAM++ tensor is not contiguous: {name}")
            member = f"{prefix}data/{tensor.storage.key}"
            raw = archive.read(member)
            storage = np.frombuffer(raw, dtype=tensor.storage.dtype)
            if storage.size != tensor.storage.size:
                raise ValueError(f"CAM++ storage size mismatch for {name}")
            count = math.prod(tensor.shape)
            end = tensor.offset + count
            if end > storage.size:
                raise ValueError(f"CAM++ tensor exceeds storage bounds: {name}")
            arrays[name] = storage[tensor.offset:end].reshape(tensor.shape).copy()
        return arrays


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required FireRedTTS3 file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _require_source_files(input_dir: Path) -> None:
    missing = [
        str(relative)
        for relative in _SOURCE_FILES.values()
        if not (input_dir / relative).is_file()
    ]
    missing.extend(
        f"text_tokenizer/{name}"
        for name in TOKENIZER_FILES
        if not (input_dir / "text_tokenizer" / name).is_file()
    )
    if missing:
        raise FileNotFoundError(
            "FireRedTTS3 source checkpoint is incomplete; missing: "
            + ", ".join(missing)
        )


def _convert_layout(component: str, name: str, value: mx.array) -> mx.array:
    if (
        component == "core"
        and name.startswith("dit.blocks.")
        and ".conv.block." in name
        and name.endswith(".weight")
    ):
        if value.ndim != 3:
            raise ValueError(
                f"expected PyTorch Conv1d weight for {name}, got {value.shape}"
            )
        return value.transpose(0, 2, 1)
    if component == "speaker" and name.endswith(".weight"):
        if value.ndim == 3:
            return value.transpose(0, 2, 1)
        if value.ndim == 4:
            return value.transpose(0, 2, 3, 1)
    return value


def _storage_dtype(component: str, name: str, value: mx.array) -> mx.Dtype:
    if value.dtype in {
        mx.int8,
        mx.int16,
        mx.int32,
        mx.int64,
        mx.uint8,
        mx.uint16,
        mx.uint32,
        mx.uint64,
    }:
        return value.dtype
    if component == "redae" and name == "decoder.istft_head.istft.window":
        return mx.float32
    if component == "speaker" and name.endswith(("running_mean", "running_var")):
        return mx.float32
    return mx.bfloat16


def convert_component_arrays(
    component: str,
    arrays: dict[str, mx.array],
) -> dict[str, mx.array]:
    """Apply the artifact's explicit MLX layout and dtype policy."""

    if component not in EXPECTED_SOURCE_TENSORS:
        raise ValueError(f"unknown FireRedTTS3 component: {component}")
    converted: dict[str, mx.array] = {}
    for name, value in arrays.items():
        layout_value = _convert_layout(component, name, value)
        converted[name] = layout_value.astype(
            _storage_dtype(component, name, layout_value)
        )
    return converted


def _save_component(
    path: Path,
    component: str,
    arrays: dict[str, mx.array],
) -> None:
    expected = EXPECTED_SOURCE_TENSORS[component]
    if len(arrays) != expected:
        raise ValueError(
            f"FireRedTTS3 {component} expected {expected} tensors, got {len(arrays)}"
        )
    converted = convert_component_arrays(component, arrays)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    mx.save_safetensors(
        temporary,
        converted,
        metadata={
            "format": "mlx",
            "model_type": MODEL_TYPE,
            "component": component,
            "precision": "bfloat16",
            "source_revision": SOURCE_REVISION,
        },
    )
    temporary.replace(path)


def build_artifact_config(input_dir: str | Path) -> FireRedTTS3Config:
    root = Path(input_dir)
    core = _load_json(root / _SOURCE_FILES["core_config"])
    redae = _load_json(root / _SOURCE_FILES["redae_config"])
    core["dtype"] = "bfloat16"
    redae["dtype"] = "bfloat16"
    payload: dict[str, Any] = {
        "model_type": MODEL_TYPE,
        "format_version": FORMAT_VERSION,
        "precision": "bfloat16",
        "source_precision": "float32",
        "files": dict(ARTIFACT_FILES),
        "tokenizer_files": list(TOKENIZER_FILES),
        "core": core,
        "redae": redae,
        "qwen": dict(_QWEN_CONFIG),
        "speaker": dict(_SPEAKER_CONFIG),
        "dtype_policy": {
            "trainable": "bfloat16",
            "float32_state": [
                "redae.decoder.istft_head.istft.window",
                "speaker.*.running_mean",
                "speaker.*.running_var",
            ],
            "integer_state": ["speaker.*.num_batches_tracked"],
        },
        "sources": {
            "weights_repo": "FireRedTeam/FireRedTTS3",
            "weights_revision": SOURCE_REVISION,
            "source_repo": "https://github.com/FireRedTeam/FireRedTTS3",
            "source_revision": SOURCE_CODE_REVISION,
        },
    }
    return FireRedTTS3Config.from_dict(payload)


def convert_fireredtts3(
    input_dir: str | Path,
    output_dir: str | Path,
) -> FireRedTTS3Config:
    """Convert the official flat bundle into one MLX BF16 artifact directory."""

    source = Path(input_dir)
    output = Path(output_dir)
    _require_source_files(source)
    output.mkdir(parents=True, exist_ok=True)

    core = mx.load(source / _SOURCE_FILES["core"])
    _save_component(output / ARTIFACT_FILES["core"], "core", core)

    redae = mx.load(source / _SOURCE_FILES["redae"])
    _save_component(output / ARTIFACT_FILES["redae"], "redae", redae)

    speaker_numpy = load_torch_zip_state_dict(source / _SOURCE_FILES["speaker"])
    speaker = {name: mx.array(value) for name, value in speaker_numpy.items()}
    _save_component(output / ARTIFACT_FILES["speaker"], "speaker", speaker)

    for name in TOKENIZER_FILES:
        shutil.copy2(source / "text_tokenizer" / name, output / name)

    config = build_artifact_config(source)
    config_path = output / "config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(
            config.to_dict(),
            handle,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        handle.write("\n")
    return config


__all__ = [
    "EXPECTED_SOURCE_TENSORS",
    "SOURCE_CODE_REVISION",
    "SOURCE_REVISION",
    "build_artifact_config",
    "convert_component_arrays",
    "convert_fireredtts3",
    "load_torch_zip_state_dict",
]
