"""Causal convolutional VAE tokenizer for VibeVoice Large.

Supports both the acoustic tokenizer (encoder + decoder, VAE with fixed σ)
and the semantic tokenizer (encoder only, deterministic).

All conv modules operate in (B, C, T) format externally, transposing at
MLX Conv1d boundaries ((B, T, C)) for faithfulness to the PyTorch reference.

Streaming inference is supported via ``VibeVoiceConvCache`` which stores
per-layer causal context between frames, matching the upstream
``VibeVoiceTokenizerStreamingCache``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from .config import VibeVoiceConvTokenizerConfig


# --------------------------------------------------------------------------- #
# Streaming cache
# --------------------------------------------------------------------------- #

@dataclass
class VibeVoiceConvCache:
    """Per-layer left-context buffers for streaming causal conv inference.

    Each conv layer registers a unique ``layer_id`` and stores its causal
    context (the last ``context_size`` samples) in this cache between calls.
    """

    buffers: dict[str, mx.array] = field(default_factory=dict)

    def get(self, layer_id: str) -> mx.array | None:
        return self.buffers.get(layer_id)

    def set(self, layer_id: str, value: mx.array) -> None:
        self.buffers[layer_id] = value

    def reset(self) -> None:
        """Zero all cached buffers (keeps keys, resets values)."""
        for key in self.buffers:
            self.buffers[key] = mx.zeros_like(self.buffers[key])

    def clear(self) -> None:
        """Remove all cached buffers."""
        self.buffers.clear()


# --------------------------------------------------------------------------- #
# ConvRMSNorm — normalizes over channel dim for (B, C, T) inputs
# --------------------------------------------------------------------------- #

class ConvRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, T) → transpose to (B, T, C) for norm → transpose back
        x = x.transpose(0, 2, 1)
        out = mx.fast.rms_norm(x.astype(mx.float32), self.weight, self.eps).astype(x.dtype)
        return out.transpose(0, 2, 1)


# --------------------------------------------------------------------------- #
# Causal Conv1d with streaming support
# --------------------------------------------------------------------------- #

_conv1d_counter = 0


def _next_conv1d_id(prefix: str) -> str:
    global _conv1d_counter
    _conv1d_counter += 1
    return f"{prefix}_{_conv1d_counter}"


class CausalConv1d(nn.Module):
    """Causal 1D convolution with manual left-padding and streaming cache.

    External format: (B, C, T). MLX Conv1d needs (B, T, C).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.context_size = (kernel_size - 1) * dilation - (stride - 1)
        self.stride = stride
        self.in_channels = in_channels
        self._layer_id = _next_conv1d_id("sconv1d")
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        if cache is not None:
            return self._forward_streaming(x, cache)
        return self._forward_non_streaming(x)

    def _forward_non_streaming(self, x: mx.array) -> mx.array:
        # x: (B, C, T)
        x = x.transpose(0, 2, 1)  # → (B, T, C)
        if self.padding > 0:
            x = mx.pad(x, [(0, 0), (self.padding, 0), (0, 0)])
        x = self.conv(x)
        return x.transpose(0, 2, 1)  # → (B, C, T)

    def _forward_streaming(self, x: mx.array, cache: VibeVoiceConvCache) -> mx.array:
        """Streaming forward: prepend cached context, conv, update cache."""
        B, C, T = x.shape

        cached = cache.get(self._layer_id)
        if cached is None:
            if self.context_size > 0:
                cached = mx.zeros((B, C, self.context_size), dtype=x.dtype)
            else:
                cached = mx.zeros((B, C, 0), dtype=x.dtype)

        # Prepend context
        if cached.shape[2] > 0:
            input_with_ctx = mx.concatenate([cached, x], axis=2)
        else:
            input_with_ctx = x

        # Conv (no extra padding in streaming mode — context IS the padding)
        h = input_with_ctx.transpose(0, 2, 1)  # (B, T+ctx, C)
        h = self.conv(h)
        h = h.transpose(0, 2, 1)  # (B, C, T_out)

        # Update cache: keep last context_size samples of input
        if self.context_size > 0:
            total = input_with_ctx.shape[2]
            if total >= self.context_size:
                new_cache = input_with_ctx[:, :, total - self.context_size :]
            else:
                new_cache = input_with_ctx
            cache.set(self._layer_id, new_cache)

        return h


# --------------------------------------------------------------------------- #
# Causal ConvTranspose1d with streaming support
# --------------------------------------------------------------------------- #

class CausalConvTranspose1d(nn.Module):
    """Causal transposed 1D convolution with right-trimming and streaming cache.

    External format: (B, C, T). MLX ConvTranspose1d needs (B, T, C).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.right_trim = kernel_size - stride
        self.stride = stride
        self.context_size = kernel_size - 1
        self._layer_id = _next_conv1d_id("sconvtr1d")
        self.convtr = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=bias,
        )

    def __call__(
        self,
        x: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        if cache is not None:
            return self._forward_streaming(x, cache)
        return self._forward_non_streaming(x)

    def _forward_non_streaming(self, x: mx.array) -> mx.array:
        # x: (B, C, T)
        x = x.transpose(0, 2, 1)  # → (B, T, C)
        x = self.convtr(x)
        x = x.transpose(0, 2, 1)  # → (B, C, T)
        if self.right_trim > 0:
            x = x[:, :, : -self.right_trim]
        return x

    def _forward_streaming(self, x: mx.array, cache: VibeVoiceConvCache) -> mx.array:
        """Streaming forward: prepend cached input, convtr, trim, return new portion."""
        B, C, T = x.shape

        cached_input = cache.get(self._layer_id)
        if cached_input is None:
            cached_input = mx.zeros((B, C, 0), dtype=x.dtype)

        # Concatenate history
        full_input = mx.concatenate([cached_input, x], axis=2)

        # Run transposed conv on full input
        h = full_input.transpose(0, 2, 1)
        h = self.convtr(h)
        h = h.transpose(0, 2, 1)  # (B, C_out, T_out)

        # Trim padding (causal: trim right only)
        if self.right_trim > 0:
            h = h[:, :, : -self.right_trim]

        # Return only the new portion (corresponding to the new input)
        if cached_input.shape[2] == 0:
            output = h
        else:
            expected_new = T * self.stride
            if h.shape[2] >= expected_new:
                output = h[:, :, -expected_new:]
            else:
                output = h

        # Cache: keep last context_size input samples
        if full_input.shape[2] > self.context_size:
            new_cache = full_input[:, :, -self.context_size :]
        else:
            new_cache = full_input
        cache.set(self._layer_id, new_cache)

        return output


# --------------------------------------------------------------------------- #
# FeedForward1D — pointwise MLP applied per time step
# --------------------------------------------------------------------------- #

class FeedForward1D(nn.Module):
    def __init__(self, dim: int, hidden: int, bias: bool = True):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden, bias=bias)
        self.linear2 = nn.Linear(hidden, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        return self.linear2(nn.gelu(self.linear1(x)))


# --------------------------------------------------------------------------- #
# Block1D — ConvNeXt-style block with streaming support
# --------------------------------------------------------------------------- #

class Block1D(nn.Module):
    """Depthwise conv mixer + FFN with per-channel layer scale."""

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        ffn_ratio: float = 4.0,
        eps: float = 1e-5,
        bias: bool = True,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.norm = ConvRMSNorm(dim, eps=eps)
        self.mixer = _DepthwiseMixer(dim, kernel_size=kernel_size, bias=bias)
        self.ffn_norm = ConvRMSNorm(dim, eps=eps)
        self.ffn = FeedForward1D(dim, int(dim * ffn_ratio), bias=bias)

        if layer_scale_init > 0:
            self.gamma = mx.ones((dim,)) * layer_scale_init
            self.ffn_gamma = mx.ones((dim,)) * layer_scale_init
        else:
            self.gamma = None
            self.ffn_gamma = None

    def __call__(
        self,
        x: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        # x: (B, C, T)

        # Mixer path
        residual = x
        h = self.norm(x)
        h = self.mixer(h, cache=cache)
        if self.gamma is not None:
            h = h * self.gamma[:, None]
        x = residual + h

        # FFN path
        residual = x
        h = self.ffn_norm(x)
        h = h.transpose(0, 2, 1)  # (B, C, T) → (B, T, C)
        h = self.ffn(h)
        h = h.transpose(0, 2, 1)  # → (B, C, T)
        if self.ffn_gamma is not None:
            h = h * self.ffn_gamma[:, None]
        x = residual + h

        return x


class _DepthwiseMixer(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, bias: bool = True):
        super().__init__()
        self.conv = _DepthwiseConvInner(dim, kernel_size=kernel_size, bias=bias)

    def __call__(
        self, x: mx.array, *, cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        return self.conv(x, cache=cache)


class _DepthwiseConvInner(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 7, bias: bool = True):
        super().__init__()
        self.conv = CausalConv1d(dim, dim, kernel_size, groups=dim, bias=bias)

    def __call__(
        self, x: mx.array, *, cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        return self.conv(x, cache=cache)


# --------------------------------------------------------------------------- #
# Wrapper modules for HF weight path matching
# --------------------------------------------------------------------------- #

class _StemConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, bias: bool = True):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, bias=bias)

    def __call__(self, x: mx.array, *, cache: VibeVoiceConvCache | None = None) -> mx.array:
        return self.conv(x, cache=cache)


class _DownsampleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, bias: bool = True):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, stride=stride, bias=bias)

    def __call__(self, x: mx.array, *, cache: VibeVoiceConvCache | None = None) -> mx.array:
        return self.conv(x, cache=cache)


class _UpsampleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, bias: bool = True):
        super().__init__()
        self.convtr = CausalConvTranspose1d(in_ch, out_ch, kernel_size, stride=stride, bias=bias)

    def __call__(self, x: mx.array, *, cache: VibeVoiceConvCache | None = None) -> mx.array:
        return self.convtr(x, cache=cache)


class _HeadConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, bias: bool = True):
        super().__init__()
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size, bias=bias)

    def __call__(self, x: mx.array, *, cache: VibeVoiceConvCache | None = None) -> mx.array:
        return self.conv(x, cache=cache)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #

class VibeVoiceConvEncoder(nn.Module):
    """Causal convolutional encoder used by both acoustic and semantic tokenizers."""

    def __init__(self, config: VibeVoiceConvTokenizerConfig):
        super().__init__()

        depths = config.parsed_encoder_depths
        ratios = list(reversed(config.encoder_ratios))
        n_filters = config.encoder_n_filters
        n_stages = len(depths)
        bias = config.conv_bias
        eps = config.layernorm_eps
        lsi = config.layer_scale_init_value

        self.downsample_layers: list[list[_StemConv | _DownsampleConv]] = []
        self.downsample_layers.append([_StemConv(config.channels, n_filters, 7, bias=bias)])
        for i in range(len(ratios)):
            in_ch = n_filters * (2 ** i)
            out_ch = n_filters * (2 ** (i + 1))
            self.downsample_layers.append([
                _DownsampleConv(in_ch, out_ch, ratios[i] * 2, ratios[i], bias=bias)
            ])

        self.stages: list[list[Block1D]] = []
        for i in range(n_stages):
            ch = n_filters * (2 ** i)
            self.stages.append([
                Block1D(ch, kernel_size=7, eps=eps, bias=bias, layer_scale_init=lsi)
                for _ in range(depths[i])
            ])

        final_ch = n_filters * (2 ** (n_stages - 1))
        self.head = _HeadConv(final_ch, config.vae_dim, kernel_size=7, bias=bias)

    def __call__(
        self,
        x: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        """Encode audio to latent space.

        Args:
            x: (B, channels, T_audio)
            cache: optional streaming cache

        Returns:
            (B, vae_dim, T_latent)
        """
        for i in range(len(self.stages)):
            x = self.downsample_layers[i][0](x, cache=cache)
            for block in self.stages[i]:
                x = block(x, cache=cache)
        return self.head(x, cache=cache)


# --------------------------------------------------------------------------- #
# Decoder
# --------------------------------------------------------------------------- #

class VibeVoiceConvDecoder(nn.Module):
    """Causal convolutional decoder for the acoustic tokenizer."""

    def __init__(self, config: VibeVoiceConvTokenizerConfig):
        super().__init__()
        self._vae_dim = config.vae_dim

        depths = config.parsed_decoder_depths
        ratios = list(config.effective_decoder_ratios)
        n_filters = config.decoder_n_filters
        n_stages = len(depths)
        bias = config.conv_bias
        eps = config.layernorm_eps
        lsi = config.layer_scale_init_value

        self.upsample_layers: list[list[_StemConv | _UpsampleConv]] = []
        max_ch = n_filters * (2 ** (n_stages - 1))
        self.upsample_layers.append([_StemConv(config.vae_dim, max_ch, 7, bias=bias)])
        for i in range(len(ratios)):
            in_ch = n_filters * (2 ** (n_stages - 1 - i))
            out_ch = n_filters * (2 ** (n_stages - 2 - i)) if i < len(ratios) - 1 else n_filters
            self.upsample_layers.append([
                _UpsampleConv(in_ch, out_ch, ratios[i] * 2, ratios[i], bias=bias)
            ])

        self.stages: list[list[Block1D]] = []
        for i in range(n_stages):
            ch = n_filters * (2 ** (n_stages - 1 - i))
            self.stages.append([
                Block1D(ch, kernel_size=7, eps=eps, bias=bias, layer_scale_init=lsi)
                for _ in range(depths[i])
            ])

        self.head = _HeadConv(n_filters, config.channels, kernel_size=7, bias=bias)

    def __call__(
        self,
        x: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        """Decode latents to audio.

        Args:
            x: (B, vae_dim, T_latent) or (B, T_latent, vae_dim)
            cache: optional streaming cache

        Returns:
            (B, channels, T_audio)
        """
        if x.ndim == 3 and x.shape[1] != self._vae_dim:
            x = x.transpose(0, 2, 1)

        for i in range(len(self.stages)):
            x = self.upsample_layers[i][0](x, cache=cache)
            for block in self.stages[i]:
                x = block(x, cache=cache)

        return self.head(x, cache=cache)


# --------------------------------------------------------------------------- #
# Acoustic Tokenizer (encoder + decoder)
# --------------------------------------------------------------------------- #

class VibeVoiceAcousticTokenizer(nn.Module):
    """Acoustic tokenizer with VAE encoder and streaming-capable decoder."""

    def __init__(self, config: VibeVoiceConvTokenizerConfig):
        super().__init__()
        self.config = config
        self.encoder = VibeVoiceConvEncoder(config)
        self.decoder = VibeVoiceConvDecoder(config)
        self.fix_std = config.fix_std
        self.std_dist_type = config.std_dist_type

    def encode(
        self,
        audio: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        """Encode audio waveform to acoustic latent mean.

        Args:
            audio: (B, 1, T_audio)
            cache: optional streaming cache for per-frame encoding

        Returns:
            mean: (B, T_latent, vae_dim) — permuted to time-first
        """
        latent = self.encoder(audio, cache=cache)
        return latent.transpose(0, 2, 1)

    def sample(self, mean: mx.array) -> mx.array:
        """Sample from the VAE distribution."""
        if self.std_dist_type == "none" or self.fix_std == 0:
            return mean
        B = mean.shape[0]
        value = self.fix_std / 0.8
        std = mx.random.normal((B,)) * value
        std = std.reshape(B, 1, 1)
        return mean + std * mx.random.normal(mean.shape)

    def decode(
        self,
        latents: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        """Decode latents to audio waveform.

        Args:
            latents: (B, T_latent, vae_dim) or (B, vae_dim, T_latent)
            cache: optional streaming cache for per-frame decoding

        Returns:
            audio: (B, 1, T_audio)
        """
        return self.decoder(latents, cache=cache)


# --------------------------------------------------------------------------- #
# Semantic Tokenizer (encoder only, deterministic)
# --------------------------------------------------------------------------- #

class VibeVoiceSemanticTokenizer(nn.Module):
    """Semantic tokenizer — encoder only, deterministic (no VAE sampling)."""

    def __init__(self, config: VibeVoiceConvTokenizerConfig):
        super().__init__()
        self.config = config
        self.encoder = VibeVoiceConvEncoder(config)

    def encode(
        self,
        audio: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        """Encode audio to semantic latent.

        Args:
            audio: (B, 1, T_audio)
            cache: optional streaming cache for per-frame encoding

        Returns:
            mean: (B, T_latent, vae_dim) — permuted to time-first
        """
        latent = self.encoder(audio, cache=cache)
        return latent.transpose(0, 2, 1)
