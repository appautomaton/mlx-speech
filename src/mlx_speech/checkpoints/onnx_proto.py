"""Minimal ONNX protobuf reader for local checkpoint inspection/loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import numpy as np


@dataclass(frozen=True)
class OnnxTensor:
    name: str
    data_type: int
    dims: tuple[int, ...]
    array: np.ndarray


@dataclass(frozen=True)
class OnnxAttribute:
    name: str
    ints: tuple[int, ...] = ()
    i: int | None = None
    f: float | None = None
    tensor: OnnxTensor | None = None


@dataclass(frozen=True)
class OnnxNode:
    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attributes: tuple[OnnxAttribute, ...]

    def get_attribute(self, name: str) -> OnnxAttribute | None:
        for attribute in self.attributes:
            if attribute.name == name:
                return attribute
        return None


@dataclass(frozen=True)
class OnnxValueInfo:
    name: str
    shape: tuple[int | str, ...]


@dataclass(frozen=True)
class LoadedOnnxGraph:
    model_path: Path
    initializers: dict[str, OnnxTensor]
    nodes: tuple[OnnxNode, ...]
    inputs: tuple[OnnxValueInfo, ...]
    outputs: tuple[OnnxValueInfo, ...]
    value_info: tuple[OnnxValueInfo, ...]


def _read_varint(buffer: bytes | memoryview, position: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = buffer[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return value, position
        shift += 7


def _skip_field(buffer: bytes | memoryview, position: int, wire_type: int) -> int:
    if wire_type == 0:
        _, position = _read_varint(buffer, position)
        return position
    if wire_type == 1:
        return position + 8
    if wire_type == 2:
        size, position = _read_varint(buffer, position)
        return position + size
    if wire_type == 5:
        return position + 4
    raise ValueError(f"Unsupported protobuf wire type: {wire_type}")


def _decode_string(buffer: bytes | memoryview, position: int) -> tuple[str, int]:
    size, position = _read_varint(buffer, position)
    text = bytes(buffer[position : position + size]).decode("utf-8", "replace")
    return text, position + size


def _parse_tensor_proto(message: bytes | memoryview) -> OnnxTensor:
    position = 0
    name = ""
    data_type = 0
    dims: list[int] = []
    raw_data: bytes | None = None
    float_data: list[float] = []
    int32_data: list[int] = []
    int64_data: list[int] = []

    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7

        if field == 1:
            if wire_type == 0:
                dim, position = _read_varint(message, position)
                dims.append(dim)
            elif wire_type == 2:
                size, position = _read_varint(message, position)
                end = position + size
                while position < end:
                    dim, position = _read_varint(message, position)
                    dims.append(dim)
            else:
                raise ValueError(f"Unsupported dims wire type: {wire_type}")
        elif field == 2 and wire_type == 0:
            data_type, position = _read_varint(message, position)
        elif field == 4:
            if wire_type == 5:
                float_data.append(
                    struct.unpack("<f", bytes(message[position : position + 4]))[0]
                )
                position += 4
            elif wire_type == 2:
                size, position = _read_varint(message, position)
                payload = bytes(message[position : position + size])
                position += size
                float_data.extend(np.frombuffer(payload, dtype="<f4").tolist())
            else:
                position = _skip_field(message, position, wire_type)
        elif field == 5:
            if wire_type == 0:
                value, position = _read_varint(message, position)
                int32_data.append(int(value))
            elif wire_type == 2:
                size, position = _read_varint(message, position)
                payload = bytes(message[position : position + size])
                position += size
                int32_data.extend(np.frombuffer(payload, dtype="<i4").tolist())
            else:
                position = _skip_field(message, position, wire_type)
        elif field == 7:
            if wire_type == 0:
                value, position = _read_varint(message, position)
                int64_data.append(int(value))
            elif wire_type == 2:
                size, position = _read_varint(message, position)
                end = position + size
                while position < end:
                    value, position = _read_varint(message, position)
                    int64_data.append(int(value))
            else:
                position = _skip_field(message, position, wire_type)
        elif field == 8 and wire_type == 2:
            name, position = _decode_string(message, position)
        elif field in {9, 10} and wire_type == 2:
            size, position = _read_varint(message, position)
            raw_data = bytes(message[position : position + size])
            position += size
        else:
            position = _skip_field(message, position, wire_type)

    if raw_data is not None:
        if data_type == 1:
            array = np.frombuffer(raw_data, dtype="<f4")
        elif data_type == 6:
            array = np.frombuffer(raw_data, dtype="<i4")
        elif data_type == 7:
            array = np.frombuffer(raw_data, dtype="<i8")
        else:
            array = np.frombuffer(raw_data, dtype=np.uint8)
    elif float_data:
        array = np.asarray(float_data, dtype=np.float32)
    elif int32_data:
        array = np.asarray(int32_data, dtype=np.int32)
    elif int64_data:
        array = np.asarray(int64_data, dtype=np.int64)
    else:
        array = np.asarray([], dtype=np.float32)

    if dims:
        array = array.reshape(tuple(int(dim) for dim in dims))

    return OnnxTensor(
        name=name,
        data_type=int(data_type),
        dims=tuple(int(dim) for dim in dims),
        array=array,
    )


def _parse_attribute_proto(message: bytes | memoryview) -> OnnxAttribute:
    position = 0
    name = ""
    ints: list[int] = []
    integer: int | None = None
    float_value: float | None = None
    tensor: OnnxTensor | None = None

    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7

        if field == 1 and wire_type == 2:
            name, position = _decode_string(message, position)
        elif field == 3 and wire_type == 5:
            float_value = struct.unpack(
                "<f", bytes(message[position : position + 4])
            )[0]
            position += 4
        elif field == 4 and wire_type == 0:
            integer, position = _read_varint(message, position)
        elif field == 5 and wire_type == 2:
            size, position = _read_varint(message, position)
            tensor = _parse_tensor_proto(message[position : position + size])
            position += size
        elif field == 8 and wire_type == 0:
            value, position = _read_varint(message, position)
            ints.append(int(value))
        elif field == 20 and wire_type == 0:
            _, position = _read_varint(message, position)
        else:
            position = _skip_field(message, position, wire_type)

    return OnnxAttribute(
        name=name,
        ints=tuple(ints),
        i=integer,
        f=float_value,
        tensor=tensor,
    )


def _parse_node_proto(message: bytes | memoryview) -> OnnxNode:
    position = 0
    inputs: list[str] = []
    outputs: list[str] = []
    name = ""
    op_type = ""
    attributes: list[OnnxAttribute] = []

    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7

        if field == 1 and wire_type == 2:
            text, position = _decode_string(message, position)
            inputs.append(text)
        elif field == 2 and wire_type == 2:
            text, position = _decode_string(message, position)
            outputs.append(text)
        elif field == 3 and wire_type == 2:
            name, position = _decode_string(message, position)
        elif field == 4 and wire_type == 2:
            op_type, position = _decode_string(message, position)
        elif field == 5 and wire_type == 2:
            size, position = _read_varint(message, position)
            attributes.append(_parse_attribute_proto(message[position : position + size]))
            position += size
        else:
            position = _skip_field(message, position, wire_type)

    return OnnxNode(
        name=name,
        op_type=op_type,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        attributes=tuple(attributes),
    )


def _parse_value_info_proto(message: bytes | memoryview) -> OnnxValueInfo:
    position = 0
    name = ""
    shape: list[int | str] = []

    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7

        if field == 1 and wire_type == 2:
            name, position = _decode_string(message, position)
        elif field == 2 and wire_type == 2:
            size, position = _read_varint(message, position)
            type_proto = message[position : position + size]
            position += size
            shape.extend(_extract_shape_from_type_proto(type_proto))
        else:
            position = _skip_field(message, position, wire_type)

    return OnnxValueInfo(name=name, shape=tuple(shape))


def _extract_shape_from_type_proto(message: bytes | memoryview) -> list[int | str]:
    position = 0
    shape: list[int | str] = []
    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7
        if field == 1 and wire_type == 2:
            size, position = _read_varint(message, position)
            tensor_type = message[position : position + size]
            position += size
            shape.extend(_extract_shape_from_tensor_type(tensor_type))
        else:
            position = _skip_field(message, position, wire_type)
    return shape


def _extract_shape_from_tensor_type(message: bytes | memoryview) -> list[int | str]:
    position = 0
    shape: list[int | str] = []
    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7
        if field == 2 and wire_type == 2:
            size, position = _read_varint(message, position)
            tensor_shape = message[position : position + size]
            position += size
            shape.extend(_extract_dims_from_shape_proto(tensor_shape))
        else:
            position = _skip_field(message, position, wire_type)
    return shape


def _extract_dims_from_shape_proto(message: bytes | memoryview) -> list[int | str]:
    position = 0
    dims: list[int | str] = []
    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7
        if field == 1 and wire_type == 2:
            size, position = _read_varint(message, position)
            dim_proto = message[position : position + size]
            position += size
            dims.append(_extract_dim_value(dim_proto))
        else:
            position = _skip_field(message, position, wire_type)
    return dims


def _extract_dim_value(message: bytes | memoryview) -> int | str:
    position = 0
    while position < len(message):
        key, position = _read_varint(message, position)
        field = key >> 3
        wire_type = key & 7
        if field == 1 and wire_type == 0:
            value, position = _read_varint(message, position)
            return int(value)
        if field == 2 and wire_type == 2:
            text, position = _decode_string(message, position)
            return text
        position = _skip_field(message, position, wire_type)
    return "?"


def load_onnx_graph(model_path: str | Path) -> LoadedOnnxGraph:
    resolved = Path(model_path)
    if not resolved.exists():
        raise FileNotFoundError(f"ONNX model not found: {resolved}")

    data = resolved.read_bytes()
    position = 0
    graph_message: memoryview | None = None

    while position < len(data):
        key, position = _read_varint(data, position)
        field = key >> 3
        wire_type = key & 7
        if field in {7, 8} and wire_type == 2:
            size, position = _read_varint(data, position)
            graph_message = memoryview(data)[position : position + size]
            break
        position = _skip_field(data, position, wire_type)

    if graph_message is None:
        raise ValueError(f"Unable to locate GraphProto inside {resolved}")

    nodes: list[OnnxNode] = []
    initializers: dict[str, OnnxTensor] = {}
    inputs: list[OnnxValueInfo] = []
    outputs: list[OnnxValueInfo] = []
    value_info: list[OnnxValueInfo] = []

    graph_position = 0
    while graph_position < len(graph_message):
        key, graph_position = _read_varint(graph_message, graph_position)
        field = key >> 3
        wire_type = key & 7

        if wire_type != 2:
            graph_position = _skip_field(graph_message, graph_position, wire_type)
            continue

        size, graph_position = _read_varint(graph_message, graph_position)
        message = graph_message[graph_position : graph_position + size]
        graph_position += size

        if field == 1:
            nodes.append(_parse_node_proto(message))
        elif field == 5:
            tensor = _parse_tensor_proto(message)
            initializers[tensor.name] = tensor
        elif field == 11:
            inputs.append(_parse_value_info_proto(message))
        elif field == 12:
            outputs.append(_parse_value_info_proto(message))
        elif field == 13:
            value_info.append(_parse_value_info_proto(message))

    return LoadedOnnxGraph(
        model_path=resolved,
        initializers=initializers,
        nodes=tuple(nodes),
        inputs=tuple(inputs),
        outputs=tuple(outputs),
        value_info=tuple(value_info),
    )


__all__ = [
    "LoadedOnnxGraph",
    "OnnxAttribute",
    "OnnxNode",
    "OnnxTensor",
    "OnnxValueInfo",
    "load_onnx_graph",
]
