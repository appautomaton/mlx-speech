"""Pure-MLX dots.tts AudioVAE encode/decode composition."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import DotsTTSVocoderConfig
from .vocoder import (
    HIGH_PRECISION_OUTPUT_TILE,
    HIGH_PRECISION_TIME_TILE,
    BigVGANDecoder,
    Conv1d,
)


DECODER_STREAM_PROJECTION_BLOCK = 16
_COMPILED_VOCODER_CACHE_LIMIT = 12
_COMPILED_VOCODER_WARM_FRAMES = (4, 16)


def _leaky_relu(value: mx.array, slope: float) -> mx.array:
    return mx.where(value >= 0, value, value * slope)


def _high_precision_matmul(value: mx.array, right: mx.array) -> mx.array:
    """Contract a 2-D right operand with true FP32 reduction."""

    value = value.astype(mx.float32)
    right = right.astype(mx.float32)
    input_features = int(value.shape[-1])
    output_features = int(right.shape[-1])
    leading_shape = tuple(int(size) for size in value.shape[:-1])
    rows = value.reshape(-1, input_features)
    row_tiles = []
    for row_start in range(0, int(rows.shape[0]), HIGH_PRECISION_TIME_TILE):
        row_end = min(row_start + HIGH_PRECISION_TIME_TILE, int(rows.shape[0]))
        output_tiles = []
        for output_start in range(
            0, output_features, HIGH_PRECISION_OUTPUT_TILE
        ):
            output_end = min(
                output_start + HIGH_PRECISION_OUTPUT_TILE, output_features
            )
            output_tiles.append(
                mx.sum(
                    rows[row_start:row_end, :, None]
                    * right[:, output_start:output_end],
                    axis=-2,
                )
            )
        row_tile = mx.concatenate(output_tiles, axis=-1)
        mx.eval(row_tile)
        row_tiles.append(row_tile)
    result = mx.concatenate(row_tiles, axis=0).reshape(
        (*leading_shape, output_features)
    )
    mx.eval(result)
    return result


def _linear(value: mx.array, layer: nn.Linear, *, high_precision: bool) -> mx.array:
    if not high_precision:
        return layer(value)
    output = _high_precision_matmul(value, layer.weight.T)
    return output if layer.bias is None else output + layer.bias.astype(mx.float32)


def _fixed_row_matmul(
    value: mx.array,
    right: mx.array,
    *,
    block_size: int,
) -> mx.array:
    """Use one row shape so batch and streamed projections round identically."""

    input_features = int(value.shape[-1])
    output_features = int(right.shape[-1])
    leading_shape = tuple(int(size) for size in value.shape[:-1])
    rows = value.reshape(-1, input_features)
    outputs = []
    for start in range(0, int(rows.shape[0]), block_size):
        block = rows[start : start + block_size]
        valid_rows = int(block.shape[0])
        if valid_rows < block_size:
            block = mx.concatenate(
                (
                    block,
                    mx.zeros(
                        (block_size - valid_rows, input_features),
                        dtype=block.dtype,
                    ),
                ),
                axis=0,
            )
        outputs.append((block @ right)[:valid_rows])
    return mx.concatenate(outputs, axis=0).reshape(
        (*leading_shape, output_features)
    )


def _stream_linear(
    value: mx.array,
    layer: nn.Linear,
    *,
    block_size: int,
) -> mx.array:
    output = _fixed_row_matmul(value, layer.weight.T, block_size=block_size)
    return output if layer.bias is None else output + layer.bias


def encoder_logical_workspace_bytes(
    config: DotsTTSVocoderConfig,
    *,
    sample_count: int,
    batch_size: int = 1,
) -> int:
    """Bound the largest explicit FP32 broadcast-reduction workspace."""

    if sample_count <= 0 or batch_size <= 0:
        raise ValueError("encoder workspace dimensions must be positive")
    largest_elements = 0

    def include(
        time: int,
        input_channels: int,
        output_channels: int,
        *,
        flattened_rows: bool = False,
    ) -> None:
        nonlocal largest_elements
        rows = (
            min(batch_size * time, HIGH_PRECISION_TIME_TILE)
            if flattened_rows
            else batch_size * min(time, HIGH_PRECISION_TIME_TILE)
        )
        largest_elements = max(
            largest_elements,
            rows * input_channels * min(output_channels, HIGH_PRECISION_OUTPUT_TILE),
        )

    time = sample_count
    include(time, 1, config.downsample_channels[0])
    for input_channels, output_channels, rate in zip(
        config.downsample_channels[:-1],
        config.downsample_channels[1:],
        config.downsample_rates,
        strict=True,
    ):
        time = (time + rate - 1) // rate
        include(time, input_channels, output_channels)
        include(time, output_channels, output_channels)
    include(time, config.downsample_channels[-1], config.latent_dim)

    intermediate = 4 * config.latent_dim
    include(time, config.latent_dim, intermediate, flattened_rows=True)
    include(time, intermediate, 4 * intermediate, flattened_rows=True)
    include(1, intermediate, 4 * intermediate, flattened_rows=True)
    include(time, intermediate, config.latent_dim, flattened_rows=True)
    include(time, config.latent_dim, 2 * config.latent_dim, flattened_rows=True)
    return largest_elements * 4  # FP32 bytes


class SLSTM(nn.Module):
    """Residual batch-first LSTM with explicit PyTorch-compatible gate order."""

    def __init__(
        self,
        dimension: int,
        num_layers: int,
        *,
        high_precision: bool = False,
        projection_block_size: int | None = None,
    ):
        super().__init__()
        self.dimension = int(dimension)
        self.layers = [_LSTMWeights(dimension) for _ in range(num_layers)]
        self.high_precision = bool(high_precision)
        self.projection_block_size = projection_block_size

    def initial_state(
        self, batch_size: int, *, dtype: mx.Dtype = mx.float32
    ) -> tuple[tuple[mx.array, mx.array], ...]:
        if batch_size <= 0:
            raise ValueError("SLSTM batch_size must be positive")
        return tuple(
            (
                mx.zeros((batch_size, self.dimension), dtype=dtype),
                mx.zeros((batch_size, self.dimension), dtype=dtype),
            )
            for _ in self.layers
        )

    def execute_chunk(
        self,
        value: mx.array,
        state: tuple[tuple[mx.array, mx.array], ...],
    ) -> tuple[mx.array, tuple[tuple[mx.array, mx.array], ...]]:
        if value.ndim != 3 or int(value.shape[-1]) != self.dimension:
            raise ValueError(
                f"SLSTM expects (batch, time, {self.dimension}), got {value.shape}"
            )
        if len(state) != len(self.layers):
            raise ValueError("SLSTM state layer count differs from the module")
        residual = value
        batch, time, _ = value.shape
        if int(time) == 0:
            return value, state
        next_state = []
        for layer_index, layer in enumerate(self.layers):
            if self.high_precision:
                projected = _high_precision_matmul(value, layer.weight_ih.T)
            elif self.projection_block_size is not None:
                projected = _fixed_row_matmul(
                    value,
                    layer.weight_ih.T,
                    block_size=self.projection_block_size,
                )
            else:
                projected = value @ layer.weight_ih.T
            projected += layer.bias_ih + layer.bias_hh
            h, cell = state[layer_index]
            expected_shape = (int(batch), self.dimension)
            if h.shape != expected_shape or cell.shape != expected_shape:
                raise ValueError(
                    "SLSTM hidden and cell state must have shape "
                    f"{expected_shape}, got {h.shape} and {cell.shape}"
                )
            outputs = []
            for index in range(int(time)):
                recurrent = (
                    _high_precision_matmul(h, layer.weight_hh.T)
                    if self.high_precision
                    else h @ layer.weight_hh.T
                )
                gates = projected[:, index] + recurrent
                input_gate, forget_gate, candidate, output_gate = mx.split(
                    gates, 4, axis=-1
                )
                input_gate = mx.sigmoid(input_gate)
                forget_gate = mx.sigmoid(forget_gate)
                candidate = mx.tanh(candidate)
                output_gate = mx.sigmoid(output_gate)
                cell = forget_gate * cell + input_gate * candidate
                h = output_gate * mx.tanh(cell)
                outputs.append(h)
            value = mx.stack(outputs, axis=1)
            next_state.append((h, cell))
        return value + residual, tuple(next_state)

    def __call__(self, value: mx.array) -> mx.array:
        state = self.initial_state(int(value.shape[0]), dtype=value.dtype)
        output, _ = self.execute_chunk(value, state)
        return output


class _LSTMWeights(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        scale = dimension**-0.5
        self.weight_ih = mx.random.normal((4 * dimension, dimension)) * scale
        self.weight_hh = mx.random.normal((4 * dimension, dimension)) * scale
        self.bias_ih = mx.zeros((4 * dimension,))
        self.bias_hh = mx.zeros((4 * dimension,))


class _ResidualStack(nn.Module):
    def __init__(self, channels: int, layers: int, *, high_precision: bool = False):
        super().__init__()
        self.convs1 = [
            Conv1d(
                channels,
                channels,
                3,
                dilation=2**index,
                causal=True,
                high_precision=high_precision,
            )
            for index in range(layers)
        ]
        self.convs2 = [
            Conv1d(
                channels, channels, 3, causal=True, high_precision=high_precision
            )
            for _ in range(layers)
        ]

    def __call__(self, value: mx.array) -> mx.array:
        for first, second in zip(self.convs1, self.convs2, strict=True):
            update = first(_leaky_relu(value, 0.01))
            value = value + second(_leaky_relu(update, 0.01))
        return value


class AudioEncoder(nn.Module):
    def __init__(
        self,
        *,
        latent_dim: int,
        channels: tuple[int, ...],
        downsample_rates: tuple[int, ...],
        residual_layers: int = 6,
        lookahead: int = 2,
    ):
        super().__init__()
        if len(channels) != len(downsample_rates) + 1:
            raise ValueError("encoder channels must be one longer than rates")
        self.pre_conv = Conv1d(
            1, channels[0], 3, causal=True, high_precision=True
        )
        self.down_convs = []
        self.residual_stacks = []
        for input_channels, output_channels, rate in zip(
            channels[:-1], channels[1:], downsample_rates, strict=True
        ):
            self.down_convs.append(
                Conv1d(
                    input_channels,
                    output_channels,
                    2 * rate,
                    stride=rate,
                    causal=True,
                    high_precision=True,
                )
            )
            self.residual_stacks.append(
                _ResidualStack(
                    output_channels, residual_layers, high_precision=True
                )
            )
        self.post_conv = Conv1d(
            channels[-1],
            latent_dim,
            2 * lookahead + 1,
            causal=False,
            high_precision=True,
        )

    def __call__(self, value: mx.array) -> mx.array:
        value = _leaky_relu(self.pre_conv(value), 0.2)
        for downsample, residual in zip(
            self.down_convs, self.residual_stacks, strict=True
        ):
            value = downsample(value)
            value = _leaky_relu(residual(value), 0.2)
        return self.post_conv(value)


class _MIBridge(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_layers: int,
        *,
        high_precision: bool = False,
        recurrent_dtype: mx.Dtype | None = None,
        projection_block_size: int | None = None,
    ):
        super().__init__()
        intermediate = 4 * latent_dim
        self.input = nn.Linear(latent_dim, intermediate, bias=True)
        self.recurrent = SLSTM(
            intermediate,
            num_layers,
            high_precision=high_precision,
            projection_block_size=projection_block_size,
        )
        self.output = nn.Linear(intermediate, latent_dim, bias=True)
        self.high_precision = bool(high_precision)
        self.recurrent_dtype = recurrent_dtype
        self.projection_block_size = projection_block_size

    def __call__(self, value: mx.array) -> mx.array:
        if self.recurrent_dtype is not None:
            value = value.astype(self.recurrent_dtype)
        value = self._project(value, self.input)
        value = self.recurrent(value)
        return self._project(value, self.output)

    def execute_chunk(
        self,
        value: mx.array,
        state: tuple[tuple[mx.array, mx.array], ...],
    ) -> tuple[mx.array, tuple[tuple[mx.array, mx.array], ...]]:
        if self.recurrent_dtype is not None:
            value = value.astype(self.recurrent_dtype)
        value = self._project(value, self.input)
        value, state = self.recurrent.execute_chunk(value, state)
        value = self._project(value, self.output)
        return value, state

    def _project(self, value: mx.array, layer: nn.Linear) -> mx.array:
        if self.projection_block_size is not None:
            return _stream_linear(
                value,
                layer,
                block_size=self.projection_block_size,
            )
        return _linear(value, layer, high_precision=self.high_precision)


@dataclass(frozen=True)
class VocoderDecodeState:
    recurrent_state: tuple[tuple[mx.array, mx.array], ...]
    decoder_input: mx.array
    maximum_chunk_size: int
    total_frames: int = 0
    emitted_frames: int = 0


@dataclass(frozen=True)
class _VocoderCompileKey:
    operation: str
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    model_identity: int


class AudioVAE(nn.Module):
    def __init__(
        self,
        config: DotsTTSVocoderConfig,
        *,
        encoder_residual_layers: int = 6,
        decoder_lookahead: int = 2,
    ):
        super().__init__()
        if config.activation != "snakebeta" or config.resblock != "1":
            raise ValueError("dots.tts AudioVAE requires snakebeta AMPBlock1")
        self.config = config
        self.latent_dim = config.latent_dim
        self.hop_size = config.hop_size
        self.decoder_lookahead = int(decoder_lookahead)
        self.audio_encoder = AudioEncoder(
            latent_dim=config.latent_dim,
            channels=config.downsample_channels,
            downsample_rates=config.downsample_rates,
            residual_layers=encoder_residual_layers,
            lookahead=2,
        )
        self.enc_mi_layer = _MIBridge(
            config.latent_dim, config.mi_num_layers, high_precision=True
        )
        self.pre_proj = Conv1d(
            config.latent_dim,
            2 * config.latent_dim,
            1,
            causal=True,
            high_precision=True,
        )
        self.post_proj = Conv1d(config.latent_dim, config.latent_dim, 1, causal=True)
        self.dec_mi_layer = _MIBridge(
            config.latent_dim,
            config.mi_num_layers,
            recurrent_dtype=mx.float32,
            projection_block_size=DECODER_STREAM_PROJECTION_BLOCK,
        )
        self.decoder = BigVGANDecoder(
            latent_dim=config.latent_dim,
            initial_channels=config.upsample_initial_channel,
            upsample_rates=config.upsample_rates,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            lookahead=self.decoder_lookahead,
        )
        self._compiled_vocoder_functions: OrderedDict[
            _VocoderCompileKey, Callable[..., tuple[mx.array, ...] | mx.array]
        ] = OrderedDict()

    def encode(self, waveform: mx.array) -> mx.array:
        if waveform.ndim != 3 or int(waveform.shape[1]) != 1:
            raise ValueError("AudioVAE encode expects waveform shape (batch, 1, samples)")
        if int(waveform.shape[-1]) < self.hop_size:
            raise ValueError("AudioVAE waveform is shorter than one latent hop")
        value = waveform.astype(mx.float32).transpose(0, 2, 1)
        value = self.audio_encoder(value)
        value = self.enc_mi_layer(value)
        return self.pre_proj(value).transpose(0, 2, 1)

    def decode(self, latent: mx.array) -> mx.array:
        if latent.ndim != 3 or int(latent.shape[1]) != self.latent_dim:
            raise ValueError(
                f"AudioVAE decode expects (batch, {self.latent_dim}, frames), "
                f"got {latent.shape}"
            )
        if int(latent.shape[-1]) <= 0:
            raise ValueError("AudioVAE decode latent must not be empty")
        value = self.post_proj(
            latent.astype(self.post_proj.weight.dtype).transpose(0, 2, 1)
        )
        value = self.dec_mi_layer(value)
        value = value.astype(self.decoder.input_dtype)
        frames = int(value.shape[1])
        decoder_padding = (
            self.decoder.stream_left_context + self.decoder.stream_lookahead
        )
        value = mx.concatenate(
            (
                value,
                mx.zeros(
                    (int(value.shape[0]), decoder_padding, self.latent_dim),
                    dtype=value.dtype,
                ),
            ),
            axis=1,
        )
        waveform = self.decoder(value)[:, : frames * self.hop_size]
        return waveform.astype(mx.float32).transpose(0, 2, 1)

    def init_decode_state(
        self,
        *,
        batch_size: int = 1,
        maximum_chunk_size: int,
    ) -> VocoderDecodeState:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        window_size = self.decoder.stream_window_size(maximum_chunk_size)
        recurrent_dtype = self.dec_mi_layer.recurrent_dtype
        if recurrent_dtype is None:
            recurrent_dtype = self.dec_mi_layer.input.weight.dtype
        decoder_dtype = self.decoder.input_dtype
        return VocoderDecodeState(
            recurrent_state=self.dec_mi_layer.recurrent.initial_state(
                batch_size, dtype=recurrent_dtype
            ),
            decoder_input=mx.zeros(
                (batch_size, window_size, self.latent_dim), dtype=decoder_dtype
            ),
            maximum_chunk_size=int(maximum_chunk_size),
        )

    @staticmethod
    def _compile_signature(
        operation: str,
        model_identity: int,
        tensors: tuple[mx.array, ...],
        execution_dtypes: tuple[mx.Dtype, ...] = (),
    ) -> _VocoderCompileKey:
        return _VocoderCompileKey(
            operation=operation,
            shapes=tuple(
                tuple(int(dimension) for dimension in tensor.shape)
                for tensor in tensors
            ),
            dtypes=(
                *(str(tensor.dtype) for tensor in tensors),
                *(str(dtype) for dtype in execution_dtypes),
            ),
            model_identity=model_identity,
        )

    def _compiled_function(
        self,
        operation: str,
        tensors: tuple[mx.array, ...],
    ) -> Callable[..., tuple[mx.array, ...] | mx.array]:
        if operation == "recurrent":
            model_identity = id(self.dec_mi_layer)
            function = self._recurrent_step_tensors
            recurrent_dtype = self.dec_mi_layer.recurrent_dtype
            if recurrent_dtype is None:
                recurrent_dtype = self.dec_mi_layer.input.weight.dtype
            execution_dtypes = (
                self.post_proj.weight.dtype,
                self.dec_mi_layer.input.weight.dtype,
                recurrent_dtype,
                self.decoder.input_dtype,
            )
            state_inputs = [self.post_proj.state, self.dec_mi_layer.state]
        elif operation == "decoder":
            model_identity = id(self.decoder)
            function = self._decode_window_tensor
            execution_dtypes = ()
            state_inputs = [self.decoder.state]
        else:
            raise ValueError(f"unsupported vocoder compile operation: {operation}")
        key = self._compile_signature(
            operation,
            model_identity,
            tensors,
            execution_dtypes,
        )
        compiled = self._compiled_vocoder_functions.pop(key, None)
        if compiled is None:
            compiled = mx.compile(function, inputs=state_inputs)
        self._compiled_vocoder_functions[key] = compiled
        while len(self._compiled_vocoder_functions) > _COMPILED_VOCODER_CACHE_LIMIT:
            eviction_key = next(
                (
                    candidate
                    for candidate in self._compiled_vocoder_functions
                    if not self._is_common_compile_key(candidate)
                ),
                next(iter(self._compiled_vocoder_functions)),
            )
            del self._compiled_vocoder_functions[eviction_key]
        return compiled

    def _is_common_compile_key(self, key: _VocoderCompileKey) -> bool:
        if key.operation == "recurrent":
            return key.shapes[0][-1] in _COMPILED_VOCODER_WARM_FRAMES
        if key.operation == "decoder":
            return key.shapes[0][1] == self.decoder.stream_window_size(16)
        return False

    def _clear_compiled_vocoder_cache(self) -> None:
        self._compiled_vocoder_functions.clear()

    def _recurrent_step_tensors(
        self,
        latent: mx.array,
        *flat_state: mx.array,
    ) -> tuple[mx.array, ...]:
        recurrent_state = tuple(
            (flat_state[index], flat_state[index + 1])
            for index in range(0, len(flat_state), 2)
        )
        value = self.post_proj(
            latent.astype(self.post_proj.weight.dtype).transpose(0, 2, 1)
        )
        decoder_chunk, recurrent_state = self.dec_mi_layer.execute_chunk(
            value, recurrent_state
        )
        return (
            decoder_chunk,
            *(tensor for layer_state in recurrent_state for tensor in layer_state),
        )

    def _execute_recurrent_step(
        self,
        latent: mx.array,
        recurrent_state: tuple[tuple[mx.array, mx.array], ...],
        *,
        use_compiled: bool,
    ) -> tuple[mx.array, tuple[tuple[mx.array, mx.array], ...]]:
        flat_state = tuple(
            tensor for layer_state in recurrent_state for tensor in layer_state
        )
        tensors = (latent, *flat_state)
        if use_compiled:
            result = self._compiled_function("recurrent", tensors)(*tensors)
        else:
            result = self._recurrent_step_tensors(*tensors)
        if not isinstance(result, tuple):
            raise TypeError("compiled vocoder recurrence returned an invalid result")
        next_flat_state = result[1:]
        next_state = tuple(
            (next_flat_state[index], next_flat_state[index + 1])
            for index in range(0, len(next_flat_state), 2)
        )
        return result[0], next_state

    def _decode_window_tensor(self, decoder_input: mx.array) -> mx.array:
        return self.decoder(decoder_input)

    def _execute_decoder_window(
        self,
        decoder_input: mx.array,
        *,
        use_compiled: bool,
    ) -> mx.array:
        if use_compiled:
            result = self._compiled_function("decoder", (decoder_input,))(
                decoder_input
            )
            if isinstance(result, tuple):
                raise TypeError("compiled BigVGAN decoder returned an invalid result")
            return result
        return self._decode_window_tensor(decoder_input)

    def decode_chunk(
        self,
        latent: mx.array,
        state: VocoderDecodeState,
        *,
        final: bool = False,
    ) -> tuple[mx.array, VocoderDecodeState]:
        return self._decode_chunk(
            latent,
            state,
            final=final,
            use_compiled=True,
        )

    def _decode_chunk(
        self,
        latent: mx.array,
        state: VocoderDecodeState,
        *,
        final: bool,
        use_compiled: bool,
    ) -> tuple[mx.array, VocoderDecodeState]:
        if latent.ndim != 3 or int(latent.shape[1]) != self.latent_dim:
            raise ValueError("AudioVAE decode chunk has invalid shape")
        if int(latent.shape[0]) != int(state.decoder_input.shape[0]):
            raise ValueError("AudioVAE decode state batch size differs from chunk")
        chunk_frames = int(latent.shape[-1])
        if chunk_frames > state.maximum_chunk_size:
            raise ValueError(
                f"AudioVAE decode chunk has {chunk_frames} frames, exceeding "
                f"maximum_chunk_size={state.maximum_chunk_size}"
            )

        recurrent_state = state.recurrent_state
        decoder_chunk = mx.zeros(
            (int(latent.shape[0]), 0, self.latent_dim),
            dtype=self.decoder.input_dtype,
        )
        if chunk_frames:
            decoder_chunk, recurrent_state = self._execute_recurrent_step(
                latent,
                recurrent_state,
                use_compiled=use_compiled,
            )

        window_size = int(state.decoder_input.shape[1])
        valid_frames = min(state.total_frames, window_size)
        combined = mx.concatenate(
            (state.decoder_input[:, :valid_frames], decoder_chunk), axis=1
        )
        if int(combined.shape[1]) > window_size:
            combined = combined[:, -window_size:]
        elif int(combined.shape[1]) < window_size:
            combined = mx.concatenate(
                (
                    combined,
                    mx.zeros(
                        (
                            int(combined.shape[0]),
                            window_size - int(combined.shape[1]),
                            self.latent_dim,
                        ),
                        dtype=combined.dtype,
                    ),
                ),
                axis=1,
            )
        combined = combined.astype(self.decoder.input_dtype)
        mx.eval(
            combined,
            *(tensor for layer_state in recurrent_state for tensor in layer_state),
        )

        total_frames = state.total_frames + chunk_frames
        stable_frames = (
            total_frames
            if final
            else max(0, total_frames - self.decoder.stream_lookahead)
        )
        next_emitted_frames = max(state.emitted_frames, stable_frames)
        next_state = VocoderDecodeState(
            recurrent_state=recurrent_state,
            decoder_input=combined,
            maximum_chunk_size=state.maximum_chunk_size,
            total_frames=total_frames,
            emitted_frames=next_emitted_frames,
        )
        if stable_frames <= state.emitted_frames:
            return (
                mx.zeros(
                    (int(latent.shape[0]), 1, 0), dtype=mx.float32
                ),
                next_state,
            )

        valid_frames = min(total_frames, window_size)
        window_start = total_frames - valid_frames
        if state.emitted_frames < window_start:
            raise RuntimeError("AudioVAE decoder window is shorter than its context")
        local_start = state.emitted_frames - window_start
        local_end = stable_frames - window_start
        waveform = self._execute_decoder_window(
            combined, use_compiled=use_compiled
        ).astype(mx.float32).transpose(0, 2, 1)
        return (
            waveform[
                :,
                :,
                local_start * self.hop_size : local_end * self.hop_size,
            ],
            next_state,
        )


__all__ = [
    "AudioEncoder",
    "AudioVAE",
    "SLSTM",
    "VocoderDecodeState",
    "encoder_logical_workspace_bytes",
]
