"""Pure-MLX dots.tts AudioVAE encode/decode composition."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import DotsTTSVocoderConfig
from .vocoder import BigVGANDecoder, Conv1d


def _leaky_relu(value: mx.array, slope: float) -> mx.array:
    return mx.where(value >= 0, value, value * slope)


class SLSTM(nn.Module):
    """Residual batch-first LSTM with explicit PyTorch-compatible gate order."""

    def __init__(self, dimension: int, num_layers: int):
        super().__init__()
        self.dimension = int(dimension)
        self.layers = [_LSTMWeights(dimension) for _ in range(num_layers)]

    def __call__(self, value: mx.array) -> mx.array:
        residual = value
        batch, time, hidden = value.shape
        for layer in self.layers:
            projected = value @ layer.weight_ih.T + layer.bias_ih + layer.bias_hh
            h = mx.zeros((batch, hidden), dtype=value.dtype)
            cell = mx.zeros((batch, hidden), dtype=value.dtype)
            outputs = []
            for index in range(int(time)):
                gates = projected[:, index] + h @ layer.weight_hh.T
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
        return value + residual


class _LSTMWeights(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        scale = dimension**-0.5
        self.weight_ih = mx.random.normal((4 * dimension, dimension)) * scale
        self.weight_hh = mx.random.normal((4 * dimension, dimension)) * scale
        self.bias_ih = mx.zeros((4 * dimension,))
        self.bias_hh = mx.zeros((4 * dimension,))


class _ResidualStack(nn.Module):
    def __init__(self, channels: int, layers: int):
        super().__init__()
        self.convs1 = [
            Conv1d(channels, channels, 3, dilation=2**index, causal=True)
            for index in range(layers)
        ]
        self.convs2 = [
            Conv1d(channels, channels, 3, causal=True) for _ in range(layers)
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
        self.pre_conv = Conv1d(1, channels[0], 3, causal=True)
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
                )
            )
            self.residual_stacks.append(
                _ResidualStack(output_channels, residual_layers)
            )
        self.post_conv = Conv1d(
            channels[-1], latent_dim, 2 * lookahead + 1, causal=False
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
    def __init__(self, latent_dim: int, num_layers: int):
        super().__init__()
        intermediate = 4 * latent_dim
        self.input = nn.Linear(latent_dim, intermediate, bias=True)
        self.recurrent = SLSTM(intermediate, num_layers)
        self.output = nn.Linear(intermediate, latent_dim, bias=True)

    def __call__(self, value: mx.array) -> mx.array:
        return self.output(self.recurrent(self.input(value)))


@dataclass(frozen=True)
class VocoderDecodeState:
    latent: mx.array
    emitted_samples: int = 0


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
        self.enc_mi_layer = _MIBridge(config.latent_dim, config.mi_num_layers)
        self.pre_proj = Conv1d(
            config.latent_dim, 2 * config.latent_dim, 1, causal=True
        )
        self.post_proj = Conv1d(config.latent_dim, config.latent_dim, 1, causal=True)
        self.dec_mi_layer = _MIBridge(config.latent_dim, config.mi_num_layers)
        self.decoder = BigVGANDecoder(
            latent_dim=config.latent_dim,
            initial_channels=config.upsample_initial_channel,
            upsample_rates=config.upsample_rates,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            resblock_kernel_sizes=config.resblock_kernel_sizes,
            resblock_dilation_sizes=config.resblock_dilation_sizes,
            lookahead=self.decoder_lookahead,
        )

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
        value = self.post_proj(latent.astype(mx.float32).transpose(0, 2, 1))
        value = self.dec_mi_layer(value)
        return self.decoder(value).transpose(0, 2, 1)

    def init_decode_state(self, *, batch_size: int = 1) -> VocoderDecodeState:
        return VocoderDecodeState(
            latent=mx.zeros((batch_size, self.latent_dim, 0), dtype=mx.float32)
        )

    def decode_chunk(
        self,
        latent: mx.array,
        state: VocoderDecodeState,
        *,
        final: bool = False,
    ) -> tuple[mx.array, VocoderDecodeState]:
        if latent.ndim != 3 or int(latent.shape[1]) != self.latent_dim:
            raise ValueError("AudioVAE decode chunk has invalid shape")
        if int(latent.shape[0]) != int(state.latent.shape[0]):
            raise ValueError("AudioVAE decode state batch size differs from chunk")
        combined = mx.concatenate((state.latent, latent.astype(mx.float32)), axis=-1)
        if int(combined.shape[-1]) == 0:
            return mx.zeros((int(combined.shape[0]), 1, 0)), state
        waveform = self.decode(combined)
        stable_frames = (
            int(combined.shape[-1])
            if final
            else max(0, int(combined.shape[-1]) - self.decoder_lookahead)
        )
        stable_samples = stable_frames * self.hop_size
        emitted = waveform[:, :, state.emitted_samples : stable_samples]
        return emitted, VocoderDecodeState(
            latent=combined,
            emitted_samples=stable_samples,
        )


__all__ = ["AudioEncoder", "AudioVAE", "SLSTM", "VocoderDecodeState"]
