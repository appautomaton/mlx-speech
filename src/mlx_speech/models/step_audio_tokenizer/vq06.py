"""Minimal MLX runtime for the Step-Audio vq06 semantic tokenizer path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ...checkpoints import LoadedOnnxGraph, load_onnx_graph
from .checkpoint import StepAudioTokenizerAssets, load_step_audio_tokenizer_assets
from .config import StepAudioVQ06Config
from .processor import StepAudioTokenizerProcessor


@dataclass(frozen=True)
class StepAudioVQ06Checkpoint:
    model_dir: Path
    config: StepAudioVQ06Config
    state_dict: dict[str, mx.array]
    graph: LoadedOnnxGraph


@dataclass(frozen=True)
class StepAudioVQ06AlignmentReport:
    missing_in_model: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return (
            not self.missing_in_model
            and not self.missing_in_checkpoint
            and not self.shape_mismatches
        )


def _linear_forward(weight: mx.array, bias: mx.array | None, x: mx.array) -> mx.array:
    y = mx.matmul(x.astype(mx.float32), weight.astype(mx.float32))
    if bias is not None:
        y = y + bias.astype(mx.float32)
    return y.astype(x.dtype)


class StepAudioVQ06LayerNorm(nn.Module):
    def __init__(self, size: int, *, eps: float):
        super().__init__()
        self.weight = mx.ones((size,), dtype=mx.float32)
        self.bias = mx.zeros((size,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        x_float = x.astype(mx.float32)
        mean = mx.mean(x_float, axis=-1, keepdims=True)
        variance = mx.mean((x_float - mean) * (x_float - mean), axis=-1, keepdims=True)
        normalized = (x_float - mean) * mx.rsqrt(variance + self.eps)
        return (normalized * self.weight + self.bias).astype(x.dtype)


class StepAudioVQ06Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, bias: bool = True):
        super().__init__()
        self.weight = mx.zeros((input_dim, output_dim), dtype=mx.float32)
        self.bias = mx.zeros((output_dim,), dtype=mx.float32) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        return _linear_forward(self.weight, self.bias, x)


class StepAudioVQ06Conv1d(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.weight = mx.zeros((output_channels, kernel_size, input_channels), dtype=mx.float32)
        self.bias = mx.zeros((output_channels,), dtype=mx.float32)
        self.stride = int(stride)
        self.padding = int(padding)

    def __call__(self, x: mx.array) -> mx.array:
        padded = mx.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)))
        y = mx.conv1d(padded.astype(mx.float32), self.weight.astype(mx.float32), stride=self.stride, padding=0)
        return (y + self.bias.astype(mx.float32)).astype(x.dtype)


class StepAudioVQ06Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        self.query = StepAudioVQ06Linear(hidden_size, hidden_size, bias=True)
        self.key = StepAudioVQ06Linear(hidden_size, hidden_size, bias=False)
        self.value = StepAudioVQ06Linear(hidden_size, hidden_size, bias=True)
        self.out = StepAudioVQ06Linear(hidden_size, hidden_size, bias=True)
        self.scale = self.head_dim ** -0.25

    def __call__(self, x: mx.array, *, attention_mask: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        q = q * self.scale
        k = k * self.scale
        scores = mx.matmul(q.astype(mx.float32), k.astype(mx.float32).transpose(0, 1, 3, 2))
        if attention_mask is not None:
            scores = scores + attention_mask.astype(mx.float32)
        weights = mx.softmax(scores, axis=-1)
        if attention_mask is not None:
            valid = attention_mask == 0
            weights = mx.where(valid, weights, mx.zeros_like(weights))
        hidden = mx.matmul(weights, v.astype(mx.float32))
        hidden = hidden.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        return self.out(hidden.astype(x.dtype))


class StepAudioVQ06MLP(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = StepAudioVQ06Linear(hidden_size, hidden_size * 4, bias=True)
        self.fc2 = StepAudioVQ06Linear(hidden_size * 4, hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class StepAudioVQ06ResidualAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, *, eps: float):
        super().__init__()
        self.attn_ln = StepAudioVQ06LayerNorm(hidden_size, eps=eps)
        self.attn = StepAudioVQ06Attention(hidden_size, num_heads)
        self.mlp_ln = StepAudioVQ06LayerNorm(hidden_size, eps=eps)
        self.mlp = StepAudioVQ06MLP(hidden_size)

    def __call__(self, x: mx.array, *, attention_mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.attn_ln(x), attention_mask=attention_mask)
        x = x + self.mlp(self.mlp_ln(x))
        return x


class StepAudioVQ06Encoder(nn.Module):
    def __init__(self, config: StepAudioVQ06Config):
        super().__init__()
        self.config = config
        self.conv1 = StepAudioVQ06Conv1d(
            config.num_mels,
            config.hidden_size,
            kernel_size=config.conv1_kernel_size,
            stride=config.conv1_stride,
            padding=config.conv1_padding,
        )
        self.conv2 = StepAudioVQ06Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.conv2_kernel_size,
            stride=config.conv2_stride,
            padding=config.conv2_padding,
        )
        self.positional_embedding = mx.zeros((config.max_positions, config.hidden_size), dtype=mx.float32)
        self.blocks = [
            StepAudioVQ06ResidualAttentionBlock(
                config.hidden_size,
                config.num_heads,
                eps=config.layer_norm_eps,
            )
            for _ in range(config.num_layers)
        ]

    def _conv_output_length(self, input_length: mx.array) -> mx.array:
        output = ((input_length + 2 * self.config.conv1_padding - self.config.conv1_kernel_size) // self.config.conv1_stride) + 1
        output = ((output + 2 * self.config.conv2_padding - self.config.conv2_kernel_size) // self.config.conv2_stride) + 1
        return output

    def _build_attention_mask(self, lengths: mx.array, max_length: int) -> mx.array:
        positions = mx.arange(max_length, dtype=mx.int32)[None, :]
        valid = positions < lengths[:, None].astype(mx.int32)
        neg_inf = mx.array(np.finfo(np.float32).min, dtype=mx.float32)
        return mx.where(valid[:, None, None, :], mx.zeros((1,), dtype=mx.float32), neg_inf)

    def __call__(self, features: mx.array, feature_lengths: mx.array) -> tuple[mx.array, mx.array]:
        x = features.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        encoded_lengths = self._conv_output_length(feature_lengths.astype(mx.int32))
        x = x + self.positional_embedding[: int(x.shape[1]), :].astype(x.dtype)
        attention_mask = self._build_attention_mask(encoded_lengths, int(x.shape[1]))
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        norms = mx.sqrt(mx.sum(x.astype(mx.float32) * x.astype(mx.float32), axis=-1, keepdims=True))
        norms = mx.maximum(norms, mx.array(self.config.l2_norm_eps, dtype=mx.float32))
        return (x.astype(mx.float32) / norms).astype(mx.float32), encoded_lengths


class StepAudioVQ06Quantizer(nn.Module):
    def __init__(self, hidden_size: int, codebook_size: int):
        super().__init__()
        self.codebook = mx.zeros((hidden_size, codebook_size), dtype=mx.float32)

    def __call__(self, features: mx.array) -> mx.array:
        features = features.astype(mx.float32)
        distances = mx.sum(features * features, axis=-1, keepdims=True)
        distances = distances - (2.0 * mx.matmul(features, self.codebook.astype(mx.float32)))
        codebook_norm = mx.sum(self.codebook.astype(mx.float32) * self.codebook.astype(mx.float32), axis=0, keepdims=True)
        distances = distances + codebook_norm
        return mx.argmax(-distances, axis=-1)


class StepAudioVQ06Model(nn.Module):
    def __init__(self, config: StepAudioVQ06Config):
        super().__init__()
        self.config = config
        self.encoder = StepAudioVQ06Encoder(config)
        self.quantizer = StepAudioVQ06Quantizer(config.hidden_size, config.codebook_size)

    def encode_features(self, features: mx.array, feature_lengths: mx.array) -> tuple[mx.array, mx.array]:
        encoded, encoded_lengths = self.encoder(features, feature_lengths)
        return self.quantizer(encoded), encoded_lengths

    def __call__(self, features: mx.array, feature_lengths: mx.array) -> tuple[mx.array, mx.array]:
        return self.encode_features(features, feature_lengths)


@dataclass(frozen=True)
class LoadedStepAudioVQ06Model:
    assets: StepAudioTokenizerAssets
    config: StepAudioVQ06Config
    checkpoint: StepAudioVQ06Checkpoint
    model: StepAudioVQ06Model
    alignment_report: StepAudioVQ06AlignmentReport
    runtime: "StepAudioVQ06Runtime"


class StepAudioVQ06Runtime:
    def __init__(
        self,
        *,
        assets: StepAudioTokenizerAssets,
        config: StepAudioVQ06Config,
        processor: StepAudioTokenizerProcessor,
        model: StepAudioVQ06Model,
    ):
        self.assets = assets
        self.config = config
        self.processor = processor
        self.model = model

    def encode_chunk(self, features: np.ndarray) -> list[int]:
        feature_length = mx.array([int(features.shape[1])], dtype=mx.int32)
        tokens, output_lengths = self.model(mx.array(features[None, :, :], dtype=mx.float32), feature_length)
        token_array = np.asarray(tokens[0, : int(output_lengths[0])].tolist(), dtype=np.int32)
        return token_array.tolist()

    def encode(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        enable_trim: bool = True,
        energy_norm: bool = True,
    ) -> list[int]:
        waveform = self.processor.preprocess_wav(
            audio,
            sample_rate,
            enable_trim=enable_trim,
            energy_norm=energy_norm,
        )
        chunks = self.processor.prepare_vq06_chunks(waveform)
        tokens: list[int] = []
        for chunk in chunks:
            tokens.extend(self.encode_chunk(chunk.features))
        return tokens


def _node_by_name(graph: LoadedOnnxGraph) -> dict[str, Any]:
    return {node.name: node for node in graph.nodes}


def _find_initializer_input(graph: LoadedOnnxGraph, node_name: str) -> str:
    node = _node_by_name(graph)[node_name]
    for input_name in node.inputs:
        if input_name in graph.initializers:
            return input_name
    raise KeyError(f"No initializer input found for ONNX node {node_name!r}")


def _find_block_count(graph: LoadedOnnxGraph) -> int:
    highest = -1
    for node in graph.nodes:
        parts = node.name.split("/")
        for part in parts:
            if part.startswith("blocks."):
                highest = max(highest, int(part.split(".")[1]))
    return highest + 1


def _node_attribute_int(graph: LoadedOnnxGraph, node_name: str, attribute_name: str, default: int) -> int:
    attribute = _node_by_name(graph)[node_name].get_attribute(attribute_name)
    if attribute is None:
        return int(default)
    if attribute.ints:
        return int(attribute.ints[0])
    if attribute.i is not None:
        return int(attribute.i)
    if attribute.tensor is not None:
        return int(attribute.tensor.array.reshape(-1)[0])
    return int(default)


def sanitize_step_audio_vq06_state_dict(graph: LoadedOnnxGraph) -> tuple[StepAudioVQ06Config, dict[str, mx.array]]:
    initializers = graph.initializers
    block_count = _find_block_count(graph)
    config = StepAudioVQ06Config(
        num_mels=int(initializers["encoders.conv1.weight"].array.shape[1]),
        hidden_size=int(initializers["encoders.conv1.weight"].array.shape[0]),
        num_heads=_node_attribute_int(graph, "/blocks.0/attn/Constant_1", "value", 20),
        num_layers=block_count,
        max_positions=int(initializers["encoders.positional_embedding"].array.shape[0]),
        codebook_size=int(initializers[_find_initializer_input(graph, "/quantizer/rq/model/layers.0/_codebook/MatMul")].array.shape[1]),
        conv1_kernel_size=_node_attribute_int(graph, "/conv1/Conv", "kernel_shape", 3),
        conv1_stride=_node_attribute_int(graph, "/conv1/Conv", "strides", 2),
        conv1_padding=_node_attribute_int(graph, "/conv1/Conv", "pads", 1),
        conv2_kernel_size=_node_attribute_int(graph, "/conv2/Conv", "kernel_shape", 3),
        conv2_stride=_node_attribute_int(graph, "/conv2/Conv", "strides", 2),
        conv2_padding=_node_attribute_int(graph, "/conv2/Conv", "pads", 1),
        l2_norm_eps=float(_node_by_name(graph)["/Constant_29"].get_attribute("value").tensor.array.reshape(-1)[0]),
    )

    state_dict: dict[str, mx.array] = {
        "encoder.positional_embedding": mx.array(initializers["encoders.positional_embedding"].array.astype(np.float32)),
        "encoder.conv1.weight": mx.array(initializers["encoders.conv1.weight"].array.transpose(0, 2, 1).astype(np.float32)),
        "encoder.conv1.bias": mx.array(initializers["encoders.conv1.bias"].array.astype(np.float32)),
        "encoder.conv2.weight": mx.array(initializers["encoders.conv2.weight"].array.transpose(0, 2, 1).astype(np.float32)),
        "encoder.conv2.bias": mx.array(initializers["encoders.conv2.bias"].array.astype(np.float32)),
        "quantizer.codebook": mx.array(
            initializers[_find_initializer_input(graph, "/quantizer/rq/model/layers.0/_codebook/MatMul")].array.astype(np.float32)
        ),
    }

    for block_index in range(block_count):
        prefix = f"encoder.blocks.{block_index}"
        state_dict[f"{prefix}.attn_ln.weight"] = mx.array(initializers[f"encoders.blocks.{block_index}.attn_ln.weight"].array.astype(np.float32))
        state_dict[f"{prefix}.attn_ln.bias"] = mx.array(initializers[f"encoders.blocks.{block_index}.attn_ln.bias"].array.astype(np.float32))
        state_dict[f"{prefix}.mlp_ln.weight"] = mx.array(initializers[f"encoders.blocks.{block_index}.mlp_ln.weight"].array.astype(np.float32))
        state_dict[f"{prefix}.mlp_ln.bias"] = mx.array(initializers[f"encoders.blocks.{block_index}.mlp_ln.bias"].array.astype(np.float32))

        query_weight = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/attn/query/MatMul")].array
        query_bias = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/attn/query/Add")].array
        key_weight = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/attn/key/MatMul")].array
        value_weight = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/attn/value/MatMul")].array
        value_bias = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/attn/value/Add")].array
        out_weight = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/attn/out/MatMul")].array
        out_bias = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/attn/out/Add")].array
        mlp_fc1_weight = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/mlp/mlp.0/MatMul")].array
        mlp_fc1_bias = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/mlp/mlp.0/Add")].array
        mlp_fc2_weight = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/mlp/mlp.2/MatMul")].array
        mlp_fc2_bias = initializers[_find_initializer_input(graph, f"/blocks.{block_index}/mlp/mlp.2/Add")].array

        state_dict[f"{prefix}.attn.query.weight"] = mx.array(query_weight.astype(np.float32))
        state_dict[f"{prefix}.attn.query.bias"] = mx.array(query_bias.astype(np.float32))
        state_dict[f"{prefix}.attn.key.weight"] = mx.array(key_weight.astype(np.float32))
        state_dict[f"{prefix}.attn.value.weight"] = mx.array(value_weight.astype(np.float32))
        state_dict[f"{prefix}.attn.value.bias"] = mx.array(value_bias.astype(np.float32))
        state_dict[f"{prefix}.attn.out.weight"] = mx.array(out_weight.astype(np.float32))
        state_dict[f"{prefix}.attn.out.bias"] = mx.array(out_bias.astype(np.float32))
        state_dict[f"{prefix}.mlp.fc1.weight"] = mx.array(mlp_fc1_weight.astype(np.float32))
        state_dict[f"{prefix}.mlp.fc1.bias"] = mx.array(mlp_fc1_bias.astype(np.float32))
        state_dict[f"{prefix}.mlp.fc2.weight"] = mx.array(mlp_fc2_weight.astype(np.float32))
        state_dict[f"{prefix}.mlp.fc2.bias"] = mx.array(mlp_fc2_bias.astype(np.float32))

    return config, state_dict


def load_step_audio_vq06_checkpoint(model_dir: str | Path | None = None) -> StepAudioVQ06Checkpoint:
    assets = load_step_audio_tokenizer_assets(model_dir)
    graph = load_onnx_graph(assets.semantic_tokenizer_path)
    config, state_dict = sanitize_step_audio_vq06_state_dict(graph)
    return StepAudioVQ06Checkpoint(
        model_dir=assets.model_dir,
        config=config,
        state_dict=state_dict,
        graph=graph,
    )


def validate_step_audio_vq06_checkpoint_against_model(
    model: StepAudioVQ06Model,
    checkpoint: StepAudioVQ06Checkpoint,
) -> StepAudioVQ06AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        checkpoint_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return StepAudioVQ06AlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_step_audio_vq06_model(
    model_dir: str | Path | None = None,
    *,
    strict: bool = True,
) -> LoadedStepAudioVQ06Model:
    assets = load_step_audio_tokenizer_assets(model_dir)
    checkpoint = load_step_audio_vq06_checkpoint(model_dir)
    model = StepAudioVQ06Model(checkpoint.config)
    report = validate_step_audio_vq06_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            "Step-Audio vq06 checkpoint/model alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only keys, "
            f"{len(report.missing_in_checkpoint)} model-only keys, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    runtime = StepAudioVQ06Runtime(
        assets=assets,
        config=checkpoint.config,
        processor=StepAudioTokenizerProcessor(assets),
        model=model,
    )
    return LoadedStepAudioVQ06Model(
        assets=assets,
        config=checkpoint.config,
        checkpoint=checkpoint,
        model=model,
        alignment_report=report,
        runtime=runtime,
    )


__all__ = [
    "LoadedStepAudioVQ06Model",
    "StepAudioVQ06AlignmentReport",
    "StepAudioVQ06Checkpoint",
    "StepAudioVQ06Config",
    "StepAudioVQ06Encoder",
    "StepAudioVQ06Model",
    "StepAudioVQ06Quantizer",
    "StepAudioVQ06ResidualAttentionBlock",
    "StepAudioVQ06Runtime",
    "load_step_audio_vq06_checkpoint",
    "load_step_audio_vq06_model",
    "sanitize_step_audio_vq06_state_dict",
    "validate_step_audio_vq06_checkpoint_against_model",
]
