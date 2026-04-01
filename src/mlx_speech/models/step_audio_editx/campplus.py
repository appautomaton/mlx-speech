"""CAMPPlus speaker-embedding support for Step-Audio CosyVoice."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ...audio import resample_audio
from ...checkpoints import LoadedOnnxGraph, OnnxNode, load_onnx_graph
from ..step_audio_tokenizer.vq02 import _kaldi_fbank
from .frontend import resolve_step_audio_cosyvoice_dir


def _node_by_name(graph: LoadedOnnxGraph) -> dict[str, OnnxNode]:
    return {node.name: node for node in graph.nodes}


def _initializer_array(graph: LoadedOnnxGraph, name: str) -> np.ndarray:
    return graph.initializers[name].array.astype(np.float32, copy=False)


def _conv_inputs(graph: LoadedOnnxGraph, node_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    node = _node_by_name(graph)[node_name]
    if len(node.inputs) < 2:
        raise ValueError(f"Conv node {node_name!r} has no weight input.")
    weight = _initializer_array(graph, node.inputs[1])
    bias = _initializer_array(graph, node.inputs[2]) if len(node.inputs) >= 3 else None
    return weight, bias


def _batchnorm_affine_inputs(graph: LoadedOnnxGraph, node_name: str) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray]:
    node = _node_by_name(graph)[node_name]
    weight = _initializer_array(graph, node.inputs[1]) if node.inputs[1] in graph.initializers else None
    bias = _initializer_array(graph, node.inputs[2]) if node.inputs[2] in graph.initializers else None
    running_mean = _initializer_array(graph, node.inputs[3])
    running_var = _initializer_array(graph, node.inputs[4])
    return weight, bias, running_mean, running_var


@dataclass(frozen=True)
class StepAudioCampPlusConfig:
    feat_dim: int = 80
    embedding_size: int = 192
    growth_rate: int = 32
    init_channels: int = 128
    block_layers: tuple[int, int, int] = (12, 24, 16)
    segment_pool_size: int = 100
    sample_rate: int = 16000
    num_mel_bins: int = 80
    output_level: str = "segment"

    @classmethod
    def from_onnx_graph(cls, graph: LoadedOnnxGraph) -> "StepAudioCampPlusConfig":
        block_counts: list[int] = []
        for block_index in range(1, 4):
            prefix = f"xvector.block{block_index}.tdnnd"
            count = sum(
                1
                for key in graph.initializers
                if key.startswith(prefix) and key.endswith(".cam_layer.linear_local.weight")
            )
            block_counts.append(count)

        dense_weight = graph.initializers["xvector.dense.linear.weight"].array
        return cls(
            feat_dim=80,
            embedding_size=int(dense_weight.shape[0]),
            growth_rate=int(graph.initializers["xvector.block1.tdnnd1.cam_layer.linear_local.weight"].array.shape[0]),
            init_channels=int(_conv_inputs(graph, "/xvector/tdnn/linear/Conv")[0].shape[0]),
            block_layers=tuple(block_counts),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class StepAudioCampPlusCheckpoint:
    model_dir: Path
    config: StepAudioCampPlusConfig
    state_dict: dict[str, mx.array]
    graph: LoadedOnnxGraph


@dataclass(frozen=True)
class StepAudioCampPlusAlignmentReport:
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


@dataclass(frozen=True)
class LoadedStepAudioCampPlusModel:
    model_dir: Path
    config: StepAudioCampPlusConfig
    checkpoint: StepAudioCampPlusCheckpoint
    model: "StepAudioCampPlusModel"
    alignment_report: StepAudioCampPlusAlignmentReport
    runtime: "StepAudioCampPlusRuntime"


class StepAudioConv1d(nn.Module):
    def __init__(self, out_channels: int, in_channels: int, kernel_size: int, *, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = True):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size, in_channels), dtype=mx.float32)
        self.bias = mx.zeros((out_channels,), dtype=mx.float32) if bias else None
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

    def __call__(self, x: mx.array) -> mx.array:
        x_nlc = x.transpose(0, 2, 1)
        y = mx.conv1d(
            x_nlc.astype(mx.float32),
            self.weight.astype(mx.float32),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.bias is not None:
            y = y + self.bias.astype(mx.float32)
        return y.transpose(0, 2, 1).astype(x.dtype)


class StepAudioConv2d(nn.Module):
    def __init__(self, out_channels: int, in_channels: int, kernel_size: tuple[int, int], *, stride: tuple[int, int] = (1, 1), padding: tuple[int, int] = (0, 0), bias: bool = True):
        super().__init__()
        self.weight = mx.zeros((out_channels, kernel_size[0], kernel_size[1], in_channels), dtype=mx.float32)
        self.bias = mx.zeros((out_channels,), dtype=mx.float32) if bias else None
        self.stride = tuple(int(v) for v in stride)
        self.padding = tuple(int(v) for v in padding)

    def __call__(self, x: mx.array) -> mx.array:
        x_nhwc = x.transpose(0, 2, 3, 1)
        y = mx.conv2d(
            x_nhwc.astype(mx.float32),
            self.weight.astype(mx.float32),
            stride=self.stride,
            padding=self.padding,
        )
        if self.bias is not None:
            y = y + self.bias.astype(mx.float32).reshape(1, 1, 1, -1)
        return y.transpose(0, 3, 1, 2).astype(x.dtype)


class StepAudioBatchNorm1d(nn.Module):
    def __init__(self, channels: int, *, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((channels,), dtype=mx.float32) if affine else None
        self.bias = mx.zeros((channels,), dtype=mx.float32) if affine else None
        self.running_mean = mx.zeros((channels,), dtype=mx.float32)
        self.running_var = mx.ones((channels,), dtype=mx.float32)
        self.eps = float(eps)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3:
            reshape = (1, -1, 1)
        elif x.ndim == 2:
            reshape = (1, -1)
        else:
            raise ValueError(f"BatchNorm1d expects rank-2 or rank-3 input, got {x.shape}.")
        y = (x.astype(mx.float32) - self.running_mean.reshape(reshape)) * mx.rsqrt(
            self.running_var.reshape(reshape) + self.eps
        )
        if self.weight is not None:
            y = y * self.weight.reshape(reshape)
        if self.bias is not None:
            y = y + self.bias.reshape(reshape)
        return y.astype(x.dtype)


class StepAudioBasicResBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, *, stride: int = 1):
        super().__init__()
        self.conv1 = StepAudioConv2d(planes, in_planes, (3, 3), stride=(stride, 1), padding=(1, 1), bias=True)
        self.conv2 = StepAudioConv2d(planes, planes, (3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.shortcut = (
            StepAudioConv2d(planes, in_planes, (1, 1), stride=(stride, 1), padding=(0, 0), bias=True)
            if stride != 1 or in_planes != planes
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        out = nn.relu(self.conv1(x))
        out = self.conv2(out)
        shortcut = x if self.shortcut is None else self.shortcut(x)
        return nn.relu(out + shortcut)


class StepAudioFCM(nn.Module):
    def __init__(self, feat_dim: int = 80, m_channels: int = 32):
        super().__init__()
        self.conv1 = StepAudioConv2d(m_channels, 1, (3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.layer1 = [
            StepAudioBasicResBlock(m_channels, m_channels, stride=2),
            StepAudioBasicResBlock(m_channels, m_channels, stride=1),
        ]
        self.layer2 = [
            StepAudioBasicResBlock(m_channels, m_channels, stride=2),
            StepAudioBasicResBlock(m_channels, m_channels, stride=1),
        ]
        self.conv2 = StepAudioConv2d(m_channels, m_channels, (3, 3), stride=(2, 1), padding=(1, 1), bias=True)
        self.out_channels = m_channels * (feat_dim // 8)

    def __call__(self, x: mx.array) -> mx.array:
        x = x[:, None, :, :]
        out = nn.relu(self.conv1(x))
        for block in self.layer1:
            out = block(out)
        for block in self.layer2:
            out = block(out)
        out = nn.relu(self.conv2(out))
        batch_size, channels, height, width = out.shape
        return out.reshape(batch_size, channels * height, width)


def _segment_average_pool(x: mx.array, *, segment_length: int) -> mx.array:
    batch_size, channels, time = x.shape
    segments: list[mx.array] = []
    for start in range(0, int(time), int(segment_length)):
        end = min(start + int(segment_length), int(time))
        segment = mx.mean(x[:, :, start:end].astype(mx.float32), axis=-1, keepdims=True)
        segments.append(mx.repeat(segment, end - start, axis=-1))
    return mx.concatenate(segments, axis=-1).astype(x.dtype)


class StepAudioCAMLayer(nn.Module):
    def __init__(
        self,
        bn_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        segment_pool_size: int,
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.linear_local = StepAudioConv1d(out_channels, bn_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.linear1 = StepAudioConv1d(bn_channels // 2, bn_channels, 1, bias=True)
        self.linear2 = StepAudioConv1d(out_channels, bn_channels // 2, 1, bias=True)
        self.segment_pool_size = int(segment_pool_size)

    def __call__(self, x: mx.array) -> mx.array:
        local = self.linear_local(x)
        context = mx.mean(x.astype(mx.float32), axis=-1, keepdims=True).astype(x.dtype)
        context = context + _segment_average_pool(x, segment_length=self.segment_pool_size)
        context = nn.relu(self.linear1(context))
        gate = mx.sigmoid(self.linear2(context).astype(mx.float32)).astype(x.dtype)
        return local * gate


class StepAudioCAMDenseTDNNLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bn_channels: int,
        kernel_size: int,
        dilation: int,
        segment_pool_size: int,
    ):
        super().__init__()
        self.nonlinear1 = StepAudioBatchNorm1d(in_channels, affine=True)
        self.linear1 = StepAudioConv1d(bn_channels, in_channels, 1, bias=True)
        self.cam_layer = StepAudioCAMLayer(
            bn_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            segment_pool_size=segment_pool_size,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.nonlinear1(x))
        x = self.linear1(x)
        x = nn.relu(x)
        return self.cam_layer(x)


class StepAudioCAMDenseTDNNBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        in_channels: int,
        *,
        out_channels: int,
        bn_channels: int,
        kernel_size: int,
        dilation: int,
        segment_pool_size: int,
    ):
        super().__init__()
        self.layers = [
            StepAudioCAMDenseTDNNLayer(
                in_channels=in_channels + index * out_channels,
                out_channels=out_channels,
                bn_channels=bn_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                segment_pool_size=segment_pool_size,
            )
            for index in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = mx.concatenate([x, layer(x)], axis=1)
        return x


class StepAudioTransitLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, bias: bool):
        super().__init__()
        self.nonlinear = StepAudioBatchNorm1d(in_channels, affine=True)
        self.linear = StepAudioConv1d(out_channels, in_channels, 1, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear(nn.relu(self.nonlinear(x)))


class StepAudioTDNNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.linear = StepAudioConv1d(out_channels, in_channels, kernel_size, stride=stride, padding=padding, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.relu(self.linear(x))


class StepAudioStatsPool(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x.astype(mx.float32), axis=-1)
        centered = x.astype(mx.float32) - mean[:, :, None]
        count = max(int(x.shape[-1]), 1)
        denom = max(count - 1, 1)
        variance = mx.sum(centered * centered, axis=-1) / float(denom)
        std = mx.sqrt(variance)
        return mx.concatenate([mean, std], axis=-1).astype(x.dtype)


class StepAudioDenseLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = StepAudioConv1d(out_channels, in_channels, 1, bias=False)
        self.nonlinear = StepAudioBatchNorm1d(out_channels, affine=False)

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            x = x[:, :, None]
            y = self.linear(x)[:, :, 0]
        else:
            y = self.linear(x)
        return self.nonlinear(y)


class StepAudioCampPlusModel(nn.Module):
    def __init__(self, config: StepAudioCampPlusConfig):
        super().__init__()
        self.config = config
        self.head = StepAudioFCM(feat_dim=config.feat_dim, m_channels=config.growth_rate)
        channels = self.head.out_channels
        self.xvector_tdnn = StepAudioTDNNLayer(channels, config.init_channels, kernel_size=5, stride=2, padding=2)
        channels = config.init_channels

        block_specs = zip(config.block_layers, (3, 3, 3), (1, 2, 2))
        self.blocks = []
        self.transits = []
        for block_index, (num_layers, kernel_size, dilation) in enumerate(block_specs):
            block = StepAudioCAMDenseTDNNBlock(
                num_layers=num_layers,
                in_channels=channels,
                out_channels=config.growth_rate,
                bn_channels=4 * config.growth_rate,
                kernel_size=kernel_size,
                dilation=dilation,
                segment_pool_size=config.segment_pool_size,
            )
            self.blocks.append(block)
            channels = channels + num_layers * config.growth_rate
            transit = StepAudioTransitLayer(
                channels,
                channels // 2,
                bias=(block_index == len(config.block_layers) - 1),
            )
            self.transits.append(transit)
            channels //= 2

        self.stats = StepAudioStatsPool()
        self.dense = StepAudioDenseLayer(channels * 2, config.embedding_size)

    def __call__(self, features: mx.array) -> mx.array:
        x = features.transpose(0, 2, 1)
        x = self.head(x)
        x = self.xvector_tdnn(x)
        for block, transit in zip(self.blocks, self.transits):
            x = transit(block(x))
        x = nn.relu(x)
        x = self.stats(x)
        return self.dense(x)


def sanitize_step_audio_campplus_state_dict(graph: LoadedOnnxGraph) -> tuple[StepAudioCampPlusConfig, dict[str, mx.array]]:
    config = StepAudioCampPlusConfig.from_onnx_graph(graph)
    state: dict[str, mx.array] = {}

    head_conv1_weight, head_conv1_bias = _conv_inputs(graph, "/head/conv1/Conv")
    state["head.conv1.weight"] = mx.array(head_conv1_weight.transpose(0, 2, 3, 1))
    state["head.conv1.bias"] = mx.array(head_conv1_bias)

    for stage_index, stage_name in enumerate(("layer1", "layer2")):
        for block_index in range(2):
            prefix = f"head.{stage_name}.{block_index}"
            base = f"/head/{stage_name}/{stage_name}.{block_index}"
            conv1_w, conv1_b = _conv_inputs(graph, f"{base}/conv1/Conv")
            conv2_w, conv2_b = _conv_inputs(graph, f"{base}/conv2/Conv")
            state[f"{prefix}.conv1.weight"] = mx.array(conv1_w.transpose(0, 2, 3, 1))
            state[f"{prefix}.conv1.bias"] = mx.array(conv1_b)
            state[f"{prefix}.conv2.weight"] = mx.array(conv2_w.transpose(0, 2, 3, 1))
            state[f"{prefix}.conv2.bias"] = mx.array(conv2_b)
            shortcut_node = f"{base}/shortcut/shortcut.0/Conv"
            if shortcut_node in _node_by_name(graph):
                shortcut_w, shortcut_b = _conv_inputs(graph, shortcut_node)
                state[f"{prefix}.shortcut.weight"] = mx.array(shortcut_w.transpose(0, 2, 3, 1))
                state[f"{prefix}.shortcut.bias"] = mx.array(shortcut_b)

    head_conv2_weight, head_conv2_bias = _conv_inputs(graph, "/head/conv2/Conv")
    state["head.conv2.weight"] = mx.array(head_conv2_weight.transpose(0, 2, 3, 1))
    state["head.conv2.bias"] = mx.array(head_conv2_bias)

    tdnn_weight, tdnn_bias = _conv_inputs(graph, "/xvector/tdnn/linear/Conv")
    state["xvector_tdnn.linear.weight"] = mx.array(tdnn_weight.transpose(0, 2, 1))
    state["xvector_tdnn.linear.bias"] = mx.array(tdnn_bias)

    for block_group_index, num_layers in enumerate(config.block_layers, start=1):
        for layer_index in range(1, num_layers + 1):
            prefix = f"blocks.{block_group_index - 1}.layers.{layer_index - 1}"
            base = f"/xvector/block{block_group_index}/tdnnd{layer_index}"
            bn_w, bn_b, bn_mean, bn_var = _batchnorm_affine_inputs(
                graph,
                f"{base}/nonlinear1/batchnorm/BatchNormalization",
            )
            state[f"{prefix}.nonlinear1.weight"] = mx.array(bn_w)
            state[f"{prefix}.nonlinear1.bias"] = mx.array(bn_b)
            state[f"{prefix}.nonlinear1.running_mean"] = mx.array(bn_mean)
            state[f"{prefix}.nonlinear1.running_var"] = mx.array(bn_var)

            linear1_w, linear1_b = _conv_inputs(graph, f"{base}/linear1/Conv")
            state[f"{prefix}.linear1.weight"] = mx.array(linear1_w.transpose(0, 2, 1))
            state[f"{prefix}.linear1.bias"] = mx.array(linear1_b)

            local_weight, _ = _conv_inputs(graph, f"{base}/cam_layer/linear_local/Conv")
            state[f"{prefix}.cam_layer.linear_local.weight"] = mx.array(local_weight.transpose(0, 2, 1))
            cam1_w, cam1_b = _conv_inputs(graph, f"{base}/cam_layer/linear1/Conv")
            cam2_w, cam2_b = _conv_inputs(graph, f"{base}/cam_layer/linear2/Conv")
            state[f"{prefix}.cam_layer.linear1.weight"] = mx.array(cam1_w.transpose(0, 2, 1))
            state[f"{prefix}.cam_layer.linear1.bias"] = mx.array(cam1_b)
            state[f"{prefix}.cam_layer.linear2.weight"] = mx.array(cam2_w.transpose(0, 2, 1))
            state[f"{prefix}.cam_layer.linear2.bias"] = mx.array(cam2_b)

    for transit_index in range(1, 4):
        prefix = f"transits.{transit_index - 1}"
        bn_w, bn_b, bn_mean, bn_var = _batchnorm_affine_inputs(
            graph,
            f"/xvector/transit{transit_index}/nonlinear/batchnorm/BatchNormalization",
        )
        state[f"{prefix}.nonlinear.weight"] = mx.array(bn_w)
        state[f"{prefix}.nonlinear.bias"] = mx.array(bn_b)
        state[f"{prefix}.nonlinear.running_mean"] = mx.array(bn_mean)
        state[f"{prefix}.nonlinear.running_var"] = mx.array(bn_var)
        linear_w, linear_b = _conv_inputs(graph, f"/xvector/transit{transit_index}/linear/Conv")
        state[f"{prefix}.linear.weight"] = mx.array(linear_w.transpose(0, 2, 1))
        if linear_b is not None:
            state[f"{prefix}.linear.bias"] = mx.array(linear_b)

    dense_w, _ = _conv_inputs(graph, "/xvector/dense/linear/Conv")
    state["dense.linear.weight"] = mx.array(dense_w.transpose(0, 2, 1))
    _, _, dense_mean, dense_var = _batchnorm_affine_inputs(
        graph,
        "/xvector/dense/nonlinear/batchnorm/BatchNormalization",
    )
    state["dense.nonlinear.running_mean"] = mx.array(dense_mean)
    state["dense.nonlinear.running_var"] = mx.array(dense_var)

    return config, state


def load_step_audio_campplus_checkpoint(model_dir: str | Path) -> StepAudioCampPlusCheckpoint:
    resolved_model_dir = Path(model_dir)
    cosyvoice_dir = resolve_step_audio_cosyvoice_dir(resolved_model_dir)
    graph = load_onnx_graph(cosyvoice_dir / "campplus.onnx")
    config, state_dict = sanitize_step_audio_campplus_state_dict(graph)
    return StepAudioCampPlusCheckpoint(
        model_dir=cosyvoice_dir,
        config=config,
        state_dict=state_dict,
        graph=graph,
    )


def validate_step_audio_campplus_checkpoint_against_model(
    model: StepAudioCampPlusModel,
    checkpoint: StepAudioCampPlusCheckpoint,
) -> StepAudioCampPlusAlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        checkpoint_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return StepAudioCampPlusAlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_step_audio_campplus_model(
    model_dir: str | Path,
    *,
    strict: bool = True,
) -> LoadedStepAudioCampPlusModel:
    checkpoint = load_step_audio_campplus_checkpoint(model_dir)
    model = StepAudioCampPlusModel(checkpoint.config)
    report = validate_step_audio_campplus_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            "Step-Audio CAMPPlus checkpoint/model alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only keys, "
            f"{len(report.missing_in_checkpoint)} model-only keys, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    runtime = StepAudioCampPlusRuntime(model=model, config=checkpoint.config)
    return LoadedStepAudioCampPlusModel(
        model_dir=checkpoint.model_dir,
        config=checkpoint.config,
        checkpoint=checkpoint,
        model=model,
        alignment_report=report,
        runtime=runtime,
    )


class StepAudioCampPlusRuntime:
    def __init__(self, *, model: StepAudioCampPlusModel, config: StepAudioCampPlusConfig):
        self.model = model
        self.config = config

    def extract_embedding(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 2:
            if waveform.shape[0] == 1:
                waveform = waveform[0]
            elif waveform.shape[1] == 1:
                waveform = waveform[:, 0]
            else:
                waveform = waveform.mean(axis=0, dtype=np.float32)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono waveform, got {waveform.shape}.")
        if sample_rate != self.config.sample_rate:
            waveform = np.asarray(
                resample_audio(
                    waveform,
                    orig_sample_rate=sample_rate,
                    target_sample_rate=self.config.sample_rate,
                ),
                dtype=np.float32,
            )

        fbank = _kaldi_fbank(
            waveform,
            sample_frequency=self.config.sample_rate,
            frame_length_ms=25.0,
            frame_shift_ms=10.0,
            num_mel_bins=self.config.num_mel_bins,
            window_type="povey",
            dither=0.0,
            remove_dc_offset=True,
            preemphasis_coefficient=0.97,
            round_to_power_of_two=True,
            snip_edges=True,
            low_freq=20.0,
            high_freq=0.0,
            rng=np.random.default_rng(seed=0),
        )
        fbank = fbank - fbank.mean(axis=0, keepdims=True)
        embedding = self.model(mx.array(fbank[None, :, :], dtype=mx.float32))
        return np.asarray(embedding, dtype=np.float32)


__all__ = [
    "LoadedStepAudioCampPlusModel",
    "StepAudioCampPlusAlignmentReport",
    "StepAudioCampPlusCheckpoint",
    "StepAudioCampPlusConfig",
    "StepAudioCampPlusModel",
    "StepAudioCampPlusRuntime",
    "load_step_audio_campplus_checkpoint",
    "load_step_audio_campplus_model",
    "sanitize_step_audio_campplus_state_dict",
    "validate_step_audio_campplus_checkpoint_against_model",
]
