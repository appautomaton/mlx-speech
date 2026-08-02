"""Causal BigVGAN primitives for dots.tts waveform decoding."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np


HIGH_PRECISION_OUTPUT_TILE = 32
HIGH_PRECISION_TIME_TILE = 512
_ALIAS_FREE_FILTER_SIZE = 12
_ALIAS_FREE_RATIO = 2
_ALIAS_FREE_OUTPUTS_PER_THREAD = 16


_alias_free_snakebeta_kernel = mx.fast.metal_kernel(
    name="dots_tts_alias_free_snakebeta",
    input_names=["inp", "up_filter", "down_filter", "alpha", "beta"],
    output_names=["out"],
    source="""
        constexpr int filter_size = 12;
        constexpr int ratio = 2;
        constexpr int outputs_per_thread = 16;
        constexpr int activation_count = 2 * outputs_per_thread + filter_size - 2;

        const int segment = int(thread_position_in_grid.x);
        const int channel = int(thread_position_in_grid.y);
        const int batch = int(thread_position_in_grid.z);
        const int length = int(inp_shape[1]);
        const int channels = int(inp_shape[2]);
        const int output_start = segment * outputs_per_thread;
        if (output_start >= length) {
            return;
        }

        const float alpha_value = metal::precise::exp(float(alpha[channel]));
        const float beta_value = metal::precise::exp(float(beta[channel]));
        float activated[activation_count];

        for (int local = 0; local < activation_count; ++local) {
            int upsample_index = ratio * output_start - (filter_size - 1) + local;
            upsample_index = max(upsample_index, 0);
            float upsampled = 0.0f;
            for (int tap = 0; tap < filter_size; ++tap) {
                const int source_offset = upsample_index - tap;
                if (source_offset >= 0 && (source_offset % ratio) == 0) {
                    const int source_time = source_offset / ratio;
                    if (source_time < length) {
                        const int input_index =
                            (batch * length + source_time) * channels + channel;
                        const int filter_index = channel * filter_size + tap;
                        upsampled +=
                            float(inp[input_index]) * float(up_filter[filter_index]);
                    }
                }
            }
            upsampled *= float(ratio);
            const float periodic = metal::precise::sin(upsampled * alpha_value);
            activated[local] =
                upsampled + periodic * periodic / (beta_value + 1.0e-9f);
        }

        const int valid_outputs = min(outputs_per_thread, length - output_start);
        for (int local = 0; local < valid_outputs; ++local) {
            float value = 0.0f;
            for (int tap = 0; tap < filter_size; ++tap) {
                const int filter_index = channel * filter_size + tap;
                value +=
                    activated[ratio * local + tap]
                    * float(down_filter[filter_index]);
            }
            const int output_index =
                (batch * length + output_start + local) * channels + channel;
            out[output_index] = T(value);
        }
    """,
)


@dataclass(frozen=True)
class SequenceStreamState:
    tail: mx.array


@dataclass(frozen=True)
class LookaheadStreamState:
    history: mx.array
    pending: mx.array


@dataclass(frozen=True)
class AMPBlockStreamState:
    activations: tuple[SequenceStreamState, ...]
    first_convs: tuple[SequenceStreamState, ...]
    second_convs: tuple[SequenceStreamState, ...]

    def arrays(self) -> tuple[mx.array, ...]:
        return tuple(
            state.tail
            for states in (self.activations, self.first_convs, self.second_convs)
            for state in states
        )


@dataclass(frozen=True)
class BigVGANStreamState:
    conv_pre: LookaheadStreamState
    upsamples: tuple[SequenceStreamState, ...]
    resblocks: tuple[tuple[AMPBlockStreamState, ...], ...]
    activation_post: SequenceStreamState
    conv_post: SequenceStreamState
    finalized: bool = False

    def arrays(self) -> tuple[mx.array, ...]:
        return (
            self.conv_pre.history,
            self.conv_pre.pending,
            *(state.tail for state in self.upsamples),
            *(
                array
                for stage in self.resblocks
                for state in stage
                for array in state.arrays()
            ),
            self.activation_post.tail,
            self.conv_post.tail,
        )


def _empty_sequence_state(
    batch_size: int,
    channels: int,
    dtype: mx.Dtype,
) -> SequenceStreamState:
    return SequenceStreamState(
        mx.zeros((batch_size, 0, channels), dtype=dtype)
    )


def _updated_tail(value: mx.array, length: int) -> mx.array:
    if length <= 0:
        return value[:, :0]
    return value[:, -min(length, int(value.shape[1])) :]


class Conv1d(nn.Module):
    """Channels-last Conv1d with explicit causal or symmetric padding."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        causal: bool = True,
        bias: bool = True,
        high_precision: bool = False,
    ):
        super().__init__()
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.kernel_size = int(kernel_size)
        self.high_precision = bool(high_precision)
        self.left_padding = self.dilation * (self.kernel_size - 1) if causal else 0
        self.padding = (
            0
            if causal
            else (self.kernel_size * self.dilation - self.dilation) // 2
        )
        scale = math.sqrt(2.0 / (input_channels * kernel_size + output_channels))
        self.weight = mx.random.normal(
            (output_channels, kernel_size, input_channels)
        ) * scale
        self.bias = mx.zeros((output_channels,)) if bias else None

    def __call__(self, value: mx.array) -> mx.array:
        if self.high_precision:
            return self._call_high_precision(value)
        if self.causal and self.left_padding:
            value = mx.pad(value, ((0, 0), (self.left_padding, 0), (0, 0)))
        output = mx.conv1d(
            value,
            self.weight,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        return output if self.bias is None else output + self.bias

    def init_stream_state(
        self,
        batch_size: int,
        *,
        dtype: mx.Dtype,
    ) -> SequenceStreamState:
        if not self.causal or self.stride != 1:
            raise ValueError("streaming Conv1d state requires causal stride one")
        return _empty_sequence_state(
            batch_size,
            int(self.weight.shape[-1]),
            dtype,
        )

    def stream(
        self,
        value: mx.array,
        state: SequenceStreamState,
    ) -> tuple[mx.array, SequenceStreamState]:
        if not self.causal or self.stride != 1:
            raise ValueError("streaming Conv1d requires causal stride one")
        frame_count = int(value.shape[1])
        if frame_count == 0:
            return (
                mx.zeros(
                    (int(value.shape[0]), 0, int(self.weight.shape[0])),
                    dtype=value.dtype,
                ),
                state,
            )
        tail_length = int(state.tail.shape[1])
        combined = mx.concatenate((state.tail, value), axis=1)
        output = self(combined)[:, tail_length : tail_length + frame_count]
        return output, SequenceStreamState(
            _updated_tail(combined, self.left_context)
        )

    def init_lookahead_state(
        self,
        batch_size: int,
        *,
        dtype: mx.Dtype,
    ) -> LookaheadStreamState:
        if self.causal or self.stride != 1:
            raise ValueError("lookahead Conv1d state requires symmetric stride one")
        channels = int(self.weight.shape[-1])
        empty = mx.zeros((batch_size, 0, channels), dtype=dtype)
        return LookaheadStreamState(history=empty, pending=empty)

    def stream_lookahead(
        self,
        value: mx.array,
        state: LookaheadStreamState,
        *,
        final: bool,
    ) -> tuple[mx.array, LookaheadStreamState]:
        if self.causal or self.stride != 1:
            raise ValueError("lookahead Conv1d requires symmetric stride one")
        available = mx.concatenate((state.pending, value), axis=1)
        available_frames = int(available.shape[1])
        stable_frames = (
            available_frames
            if final
            else max(0, available_frames - self.right_context)
        )
        if stable_frames == 0:
            return (
                mx.zeros(
                    (int(value.shape[0]), 0, int(self.weight.shape[0])),
                    dtype=value.dtype,
                ),
                LookaheadStreamState(
                    history=state.history,
                    pending=available,
                ),
            )
        history_frames = int(state.history.shape[1])
        combined = mx.concatenate((state.history, available), axis=1)
        output = self(combined)[
            :, history_frames : history_frames + stable_frames
        ]
        processed = available[:, :stable_frames]
        history = _updated_tail(
            mx.concatenate((state.history, processed), axis=1),
            self.left_context,
        )
        return output, LookaheadStreamState(
            history=history,
            pending=available[:, stable_frames:],
        )

    @property
    def left_context(self) -> int:
        if self.causal:
            return self.left_padding
        return self.padding

    @property
    def right_context(self) -> int:
        if self.causal:
            return 0
        receptive_field = self.dilation * (self.kernel_size - 1)
        return receptive_field - self.padding

    def _call_high_precision(self, value: mx.array) -> mx.array:
        """Run an encoder-only convolution with true FP32 reductions."""

        value = value.astype(mx.float32)
        if self.causal and self.left_padding:
            value = mx.pad(value, ((0, 0), (self.left_padding, 0), (0, 0)))
        elif self.padding:
            value = mx.pad(value, ((0, 0), (self.padding, self.padding), (0, 0)))
        weight = self.weight.astype(mx.float32)
        kernel = int(weight.shape[1])
        padded_time = int(value.shape[1])
        output_time = (
            padded_time - self.dilation * (kernel - 1) - 1
        ) // self.stride + 1
        time_tiles = []
        output_channels = int(weight.shape[0])
        for time_start in range(0, output_time, HIGH_PRECISION_TIME_TILE):
            time_end = min(time_start + HIGH_PRECISION_TIME_TILE, output_time)
            channel_tiles = []
            for channel_start in range(
                0, output_channels, HIGH_PRECISION_OUTPUT_TILE
            ):
                channel_end = min(
                    channel_start + HIGH_PRECISION_OUTPUT_TILE, output_channels
                )
                output = None
                for tap in range(kernel):
                    start = tap * self.dilation + time_start * self.stride
                    frames = value[
                        :,
                        start : start + self.stride * (time_end - time_start) : self.stride,
                        :,
                    ]
                    contribution = mx.sum(
                        frames[:, :, None, :]
                        * weight[
                            None,
                            None,
                            channel_start:channel_end,
                            tap,
                            :,
                        ],
                        axis=-1,
                    )
                    output = (
                        contribution if output is None else output + contribution
                    )
                if output is None:
                    raise ValueError("high-precision convolution has an empty kernel")
                if self.bias is not None:
                    output += self.bias[channel_start:channel_end].astype(mx.float32)
                channel_tiles.append(output)
            time_tile = mx.concatenate(channel_tiles, axis=-1)
            mx.eval(time_tile)
            time_tiles.append(time_tile)
        result = mx.concatenate(time_tiles, axis=1)
        mx.eval(result)
        return result


class CausalConvTranspose1d(nn.Module):
    """MLX-layout causal transposed convolution with exact stride trimming."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        *,
        stride: int,
        bias: bool = True,
    ):
        super().__init__()
        if kernel_size != 2 * stride:
            raise ValueError("causal transposed convolution requires kernel_size=2*stride")
        self.stride = int(stride)
        self.kernel_size = int(kernel_size)
        scale = math.sqrt(2.0 / (input_channels * kernel_size + output_channels))
        self.weight = mx.random.normal(
            (output_channels, kernel_size, input_channels)
        ) * scale
        self.bias = mx.zeros((output_channels,)) if bias else None

    def __call__(self, value: mx.array) -> mx.array:
        output = mx.conv_transpose1d(value, self.weight, stride=self.stride)
        if self.bias is not None:
            output += self.bias
        return output[:, : -self.stride]

    def init_stream_state(
        self,
        batch_size: int,
        *,
        dtype: mx.Dtype,
    ) -> SequenceStreamState:
        return _empty_sequence_state(
            batch_size,
            int(self.weight.shape[-1]),
            dtype,
        )

    def stream(
        self,
        value: mx.array,
        state: SequenceStreamState,
    ) -> tuple[mx.array, SequenceStreamState]:
        frame_count = int(value.shape[1])
        if frame_count == 0:
            return (
                mx.zeros(
                    (int(value.shape[0]), 0, int(self.weight.shape[0])),
                    dtype=value.dtype,
                ),
                state,
            )
        tail_length = int(state.tail.shape[1])
        combined = mx.concatenate((state.tail, value), axis=1)
        output = self(combined)[
            :,
            tail_length * self.stride : (tail_length + frame_count)
            * self.stride,
        ]
        return output, SequenceStreamState(
            _updated_tail(combined, self.left_context)
        )

    @property
    def left_context(self) -> int:
        overlap = self.kernel_size - self.stride
        return (overlap + self.stride - 1) // self.stride


def _default_filter(kernel_size: int, ratio: int) -> np.ndarray:
    position = np.arange(kernel_size, dtype=np.float64) - (kernel_size - 1) / 2
    cutoff = 0.5 / ratio
    sinc = 2 * cutoff * np.sinc(2 * cutoff * position)
    window = np.kaiser(kernel_size, beta=8.6)
    taps = sinc * window
    taps /= taps.sum()
    return taps.astype(np.float32)


class AliasFreeSnakeBeta(nn.Module):
    """Upsample → SnakeBeta → low-pass downsample alias-free activation."""

    def __init__(
        self,
        channels: int,
        *,
        ratio: int = 2,
        kernel_size: int = 12,
    ):
        super().__init__()
        if ratio <= 0 or kernel_size <= ratio:
            raise ValueError("alias-free resampler dimensions are invalid")
        self.ratio = int(ratio)
        taps = mx.array(_default_filter(kernel_size, ratio))
        self.up_filter = mx.broadcast_to(
            taps[None, :, None], (channels, kernel_size, 1)
        )
        self.down_filter = mx.broadcast_to(
            taps[None, :, None], (channels, kernel_size, 1)
        )
        self.alpha = mx.zeros((channels,))
        self.beta = mx.zeros((channels,))

    def __call__(self, value: mx.array) -> mx.array:
        if (
            self.ratio == _ALIAS_FREE_RATIO
            and int(self.up_filter.shape[1]) == _ALIAS_FREE_FILTER_SIZE
            and int(self.down_filter.shape[1]) == _ALIAS_FREE_FILTER_SIZE
            and value.dtype in (mx.float16, mx.bfloat16)
            and mx.metal.is_available()
        ):
            segment_count = (
                int(value.shape[1]) + _ALIAS_FREE_OUTPUTS_PER_THREAD - 1
            ) // _ALIAS_FREE_OUTPUTS_PER_THREAD
            return _alias_free_snakebeta_kernel(
                inputs=[
                    value,
                    self.up_filter,
                    self.down_filter,
                    self.alpha,
                    self.beta,
                ],
                grid=(segment_count, int(value.shape[-1]), int(value.shape[0])),
                threadgroup=(128, 1, 1),
                template=[("T", value.dtype)],
                output_shapes=[value.shape],
                output_dtypes=[value.dtype],
            )[0]
        return self._eager(value)

    def _eager(self, value: mx.array) -> mx.array:
        channels = int(value.shape[-1])
        upsampled = self.ratio * mx.conv_transpose1d(
            value.astype(mx.float32),
            self.up_filter,
            stride=self.ratio,
            groups=channels,
        )
        trim = int(self.up_filter.shape[1]) - self.ratio
        upsampled = upsampled[:, :-trim]
        alpha = mx.exp(self.alpha.astype(mx.float32))[None, None]
        beta = mx.exp(self.beta.astype(mx.float32))[None, None]
        activated = upsampled + mx.square(mx.sin(upsampled * alpha)) / (beta + 1e-9)
        left = int(self.down_filter.shape[1]) - 1
        padded = mx.concatenate(
            (
                mx.broadcast_to(
                    activated[:, :1],
                    (int(activated.shape[0]), left, channels),
                ),
                activated,
            ),
            axis=1,
        )
        return mx.conv1d(
            padded,
            self.down_filter,
            stride=self.ratio,
            groups=channels,
        ).astype(value.dtype)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        dtype: mx.Dtype,
    ) -> SequenceStreamState:
        return _empty_sequence_state(
            batch_size,
            int(self.alpha.shape[0]),
            dtype,
        )

    def stream(
        self,
        value: mx.array,
        state: SequenceStreamState,
    ) -> tuple[mx.array, SequenceStreamState]:
        frame_count = int(value.shape[1])
        if frame_count == 0:
            return value, state
        tail_length = int(state.tail.shape[1])
        combined = mx.concatenate((state.tail, value), axis=1)
        output = self(combined)[:, tail_length : tail_length + frame_count]
        return output, SequenceStreamState(
            _updated_tail(combined, self.left_context)
        )

    @property
    def left_context(self) -> int:
        upsample_context = int(self.up_filter.shape[1]) - 1
        downsample_context = int(self.down_filter.shape[1]) - 1
        return (
            upsample_context + downsample_context + self.ratio - 1
        ) // self.ratio


class AMPBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ):
        super().__init__()
        self.convs1 = [
            Conv1d(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                causal=True,
            )
            for dilation in dilations
        ]
        self.convs2 = [
            Conv1d(channels, channels, kernel_size, causal=True)
            for _ in dilations
        ]
        self.activations = [
            AliasFreeSnakeBeta(channels) for _ in range(2 * len(dilations))
        ]

    def __call__(self, value: mx.array) -> mx.array:
        for index, (first, second) in enumerate(
            zip(self.convs1, self.convs2, strict=True)
        ):
            update = self.activations[2 * index](value)
            update = first(update)
            update = self.activations[2 * index + 1](update)
            value = value + second(update)
        return value

    def init_stream_state(
        self,
        batch_size: int,
        *,
        dtype: mx.Dtype,
    ) -> AMPBlockStreamState:
        return AMPBlockStreamState(
            activations=tuple(
                activation.init_stream_state(batch_size, dtype=dtype)
                for activation in self.activations
            ),
            first_convs=tuple(
                conv.init_stream_state(batch_size, dtype=dtype)
                for conv in self.convs1
            ),
            second_convs=tuple(
                conv.init_stream_state(batch_size, dtype=dtype)
                for conv in self.convs2
            ),
        )

    def stream(
        self,
        value: mx.array,
        state: AMPBlockStreamState,
    ) -> tuple[mx.array, AMPBlockStreamState]:
        activation_states = list(state.activations)
        first_conv_states = list(state.first_convs)
        second_conv_states = list(state.second_convs)
        for index, (first, second) in enumerate(
            zip(self.convs1, self.convs2, strict=True)
        ):
            update, activation_states[2 * index] = self.activations[
                2 * index
            ].stream(value, activation_states[2 * index])
            update, first_conv_states[index] = first.stream(
                update, first_conv_states[index]
            )
            update, activation_states[2 * index + 1] = self.activations[
                2 * index + 1
            ].stream(update, activation_states[2 * index + 1])
            update, second_conv_states[index] = second.stream(
                update, second_conv_states[index]
            )
            value = value + update
        return value, AMPBlockStreamState(
            activations=tuple(activation_states),
            first_convs=tuple(first_conv_states),
            second_convs=tuple(second_conv_states),
        )

    @property
    def left_context(self) -> int:
        context = 0
        for index, (first, second) in enumerate(
            zip(self.convs1, self.convs2, strict=True)
        ):
            context += self.activations[2 * index].left_context
            context += first.left_context
            context += self.activations[2 * index + 1].left_context
            context += second.left_context
        return context


class BigVGANDecoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        initial_channels: int,
        upsample_rates: tuple[int, ...],
        upsample_kernel_sizes: tuple[int, ...],
        resblock_kernel_sizes: tuple[int, ...],
        resblock_dilation_sizes: tuple[tuple[int, ...], ...],
        lookahead: int = 2,
    ):
        super().__init__()
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError("decoder upsample rates and kernels differ in length")
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError("decoder residual kernels and dilations differ in length")
        self.lookahead = int(lookahead)
        self.conv_pre = Conv1d(
            latent_dim,
            initial_channels,
            2 * self.lookahead + 1,
            causal=False,
        )
        self.ups = []
        self.resblocks = []
        channels = int(initial_channels)
        for rate, kernel in zip(
            upsample_rates, upsample_kernel_sizes, strict=True
        ):
            output_channels = channels // 2
            self.ups.append(
                CausalConvTranspose1d(
                    channels,
                    output_channels,
                    kernel,
                    stride=rate,
                )
            )
            self.resblocks.append(
                [
                    AMPBlock(output_channels, residual_kernel, dilations)
                    for residual_kernel, dilations in zip(
                        resblock_kernel_sizes,
                        resblock_dilation_sizes,
                        strict=True,
                    )
                ]
            )
            channels = output_channels
        self.activation_post = AliasFreeSnakeBeta(channels)
        self.conv_post = Conv1d(channels, 1, 7, causal=True, bias=False)

    def __call__(self, value: mx.array) -> mx.array:
        if value.dtype != self.input_dtype:
            raise ValueError(
                "BigVGAN decoder input dtype must match conv_pre weights: "
                f"expected {self.input_dtype}, got {value.dtype}"
            )
        value = self.conv_pre(value)
        for upsample, blocks in zip(self.ups, self.resblocks, strict=True):
            value = upsample(value)
            outputs = [block(value) for block in blocks]
            value = sum(outputs[1:], outputs[0]) / len(outputs)
        return mx.clip(self.conv_post(self.activation_post(value)), -1.0, 1.0)

    def init_stream_state(
        self,
        batch_size: int,
        *,
        dtype: mx.Dtype | None = None,
    ) -> BigVGANStreamState:
        stream_dtype = self.input_dtype if dtype is None else dtype
        return BigVGANStreamState(
            conv_pre=self.conv_pre.init_lookahead_state(
                batch_size,
                dtype=stream_dtype,
            ),
            upsamples=tuple(
                upsample.init_stream_state(batch_size, dtype=stream_dtype)
                for upsample in self.ups
            ),
            resblocks=tuple(
                tuple(
                    block.init_stream_state(batch_size, dtype=stream_dtype)
                    for block in blocks
                )
                for blocks in self.resblocks
            ),
            activation_post=self.activation_post.init_stream_state(
                batch_size,
                dtype=stream_dtype,
            ),
            conv_post=self.conv_post.init_stream_state(
                batch_size,
                dtype=stream_dtype,
            ),
        )

    def stream(
        self,
        value: mx.array,
        state: BigVGANStreamState,
        *,
        final: bool = False,
    ) -> tuple[mx.array, BigVGANStreamState]:
        if value.dtype != self.input_dtype:
            raise ValueError(
                "BigVGAN decoder input dtype must match conv_pre weights: "
                f"expected {self.input_dtype}, got {value.dtype}"
            )
        if int(value.shape[0]) != int(state.conv_pre.history.shape[0]):
            raise ValueError("BigVGAN stream batch size changed")
        if state.finalized:
            if int(value.shape[1]) == 0:
                return mx.zeros(
                    (int(value.shape[0]), 0, 1), dtype=value.dtype
                ), state
            raise ValueError("BigVGAN stream is already finalized")

        value, conv_pre_state = self.conv_pre.stream_lookahead(
            value,
            state.conv_pre,
            final=final,
        )
        if int(value.shape[1]) == 0:
            return mx.zeros(
                (int(value.shape[0]), 0, 1), dtype=value.dtype
            ), BigVGANStreamState(
                conv_pre=conv_pre_state,
                upsamples=state.upsamples,
                resblocks=state.resblocks,
                activation_post=state.activation_post,
                conv_post=state.conv_post,
                finalized=final,
            )

        upsample_states = list(state.upsamples)
        resblock_states = [list(stage) for stage in state.resblocks]
        for stage_index, (upsample, blocks) in enumerate(
            zip(self.ups, self.resblocks, strict=True)
        ):
            value, upsample_states[stage_index] = upsample.stream(
                value,
                upsample_states[stage_index],
            )
            outputs = []
            for block_index, block in enumerate(blocks):
                output, resblock_states[stage_index][block_index] = block.stream(
                    value,
                    resblock_states[stage_index][block_index],
                )
                outputs.append(output)
            value = sum(outputs[1:], outputs[0]) / len(outputs)

        value, activation_post_state = self.activation_post.stream(
            value,
            state.activation_post,
        )
        value, conv_post_state = self.conv_post.stream(
            value,
            state.conv_post,
        )
        return mx.clip(value, -1.0, 1.0), BigVGANStreamState(
            conv_pre=conv_pre_state,
            upsamples=tuple(upsample_states),
            resblocks=tuple(tuple(stage) for stage in resblock_states),
            activation_post=activation_post_state,
            conv_post=conv_post_state,
            finalized=final,
        )

    @property
    def input_dtype(self) -> mx.Dtype:
        return self.conv_pre.weight.dtype

    @property
    def stream_lookahead(self) -> int:
        return self.conv_pre.right_context


__all__ = [
    "AliasFreeSnakeBeta",
    "AMPBlock",
    "AMPBlockStreamState",
    "BigVGANDecoder",
    "BigVGANStreamState",
    "CausalConvTranspose1d",
    "Conv1d",
    "HIGH_PRECISION_OUTPUT_TILE",
    "HIGH_PRECISION_TIME_TILE",
    "LookaheadStreamState",
    "SequenceStreamState",
]
