"""MLX Cat audio tokenizer modules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModuleConfig,
    MossAudioTokenizerQuantizerConfig,
)

MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE = mx.bfloat16


def _make_additive_causal_mask(
    seq_len: int,
    *,
    dtype: mx.Dtype,
    context: int | None = None,
) -> mx.array:
    positions = mx.arange(seq_len)
    delta = positions[:, None] - positions[None, :]
    valid = delta >= 0
    if context is not None:
        valid = mx.logical_and(valid, delta < context)
    zero = mx.array(0, dtype=dtype)
    neg_inf = mx.array(mx.finfo(dtype).min, dtype=dtype)
    return mx.where(valid[None, None, :, :], zero, neg_inf)


def _create_sinusoidal_embedding(
    seq_len: int,
    dim: int,
    *,
    max_period: float,
    dtype: mx.Dtype,
) -> mx.array:
    if dim % 2 != 0:
        raise ValueError(f"Sinusoidal embedding requires even dim, got {dim}.")
    half_dim = dim // 2
    positions = mx.arange(seq_len, dtype=mx.float32)[:, None]
    freq_idx = mx.arange(half_dim, dtype=mx.float32)[None, :]
    denom = mx.power(mx.array(max_period, dtype=mx.float32), freq_idx / max(half_dim - 1, 1))
    phase = positions / denom
    embedding = mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)
    return embedding.astype(dtype)[None, :, :]


def _apply_codec_rope(
    q: mx.array,
    k: mx.array,
    *,
    offset: int = 0,
    max_period: float = 10_000.0,
) -> tuple[mx.array, mx.array]:
    """Match upstream Cat codec RoPE exactly.

    Upstream treats adjacent pairs as complex values:
    `[..., D] -> [..., D/2, 2]`, rotates them, then reshapes back.
    """

    if q.shape != k.shape:
        raise ValueError(f"Expected q.shape == k.shape, got {q.shape} vs {k.shape}.")
    head_dim = int(q.shape[-1])
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"Codec RoPE requires an even head_dim, got {head_dim}.")

    batch_size, num_heads, seq_len, _ = q.shape
    dtype = q.dtype

    ds = mx.arange(head_dim // 2, dtype=mx.float32)
    freqs = mx.exp(ds * (-math.log(max_period) * 2.0 / head_dim))
    positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
    phase = positions[None, None, :, None] * freqs[None, None, None, :]

    q_pairs = q.astype(mx.float32).reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)
    k_pairs = k.astype(mx.float32).reshape(batch_size, num_heads, seq_len, head_dim // 2, 2)

    qr = q_pairs[..., 0]
    qi = q_pairs[..., 1]
    kr = k_pairs[..., 0]
    ki = k_pairs[..., 1]

    rotr = mx.cos(phase)
    roti = mx.sin(phase)

    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr

    q_out = mx.stack([qor, qoi], axis=-1).reshape(batch_size, num_heads, seq_len, head_dim)
    k_out = mx.stack([kor, koi], axis=-1).reshape(batch_size, num_heads, seq_len, head_dim)
    return q_out.astype(dtype), k_out.astype(dtype)


class MossAudioTokenizerLayerScale(nn.Module):
    """Per-channel layer scale used in the codec transformer blocks."""

    def __init__(self, channels: int, init: float = 1e-4):
        super().__init__()
        self.scale = mx.full((channels,), init)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


class MossAudioTokenizerPointwiseConv1d(nn.Module):
    """Pointwise 1x1 conv implemented as a per-time linear projection."""

    def __init__(self, input_dim: int, output_dim: int, *, bias: bool = True):
        super().__init__()
        scale = math.sqrt(1.0 / max(input_dim, 1))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(output_dim, input_dim, 1),
        )
        if bias:
            self.bias = mx.zeros((output_dim,))

    def __call__(self, x: mx.array) -> mx.array:
        y = x.transpose(0, 2, 1) @ self.weight[:, :, 0].T
        if "bias" in self:
            y = y + self.bias
        return y.transpose(0, 2, 1)


class MossAudioTokenizerMultiheadAttention(nn.Module):
    """Minimal non-streaming causal multi-head attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        causal: bool = True,
        context: int | None = None,
        use_rope: bool = False,
        max_period: float = 10_000.0,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.context = context
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.in_projs = [nn.Linear(embed_dim, 3 * embed_dim, bias=False)]
        self.out_projs = [nn.Linear(embed_dim, embed_dim, bias=False)]
        self.use_rope = use_rope
        self.max_period = max_period

    def __call__(self, x: mx.array) -> mx.array:
        batch_size, seq_len, _ = x.shape
        projected = self.in_projs[0](x).astype(x.dtype)
        projected = projected.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        projected = projected.transpose(2, 0, 3, 1, 4)
        query, key, value = projected[0], projected[1], projected[2]

        if self.use_rope:
            query, key = _apply_codec_rope(
                query,
                key,
                offset=0,
                max_period=self.max_period,
            )

        mask = None
        if self.causal:
            mask = _make_additive_causal_mask(
                seq_len,
                dtype=x.dtype,
                context=self.context,
            )

        attn_output = mx.fast.scaled_dot_product_attention(
            query,
            key,
            value,
            scale=self.scale,
            mask=mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        return self.out_projs[0](attn_output).astype(x.dtype)


class MossAudioTokenizerTransformerLayer(nn.Module):
    """One non-streaming codec transformer layer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dim_feedforward: int,
        causal: bool,
        context: int | None,
        positional_embedding: str,
        max_period: float,
        layer_scale: float | None,
    ):
        super().__init__()
        use_rope = positional_embedding in {"rope", "sin_rope"}
        self.self_attn = MossAudioTokenizerMultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            causal=causal,
            context=context,
            use_rope=use_rope,
            max_period=max_period,
        )
        self.norm1 = nn.LayerNorm(d_model, eps=1e-5, affine=True, bias=True)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5, affine=True, bias=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)
        self.layer_scale_1 = (
            MossAudioTokenizerLayerScale(d_model, init=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.layer_scale_2 = (
            MossAudioTokenizerLayerScale(d_model, init=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )

    def __call__(self, x: mx.array) -> mx.array:
        x_dtype = x.dtype
        attn_update = self.self_attn(self.norm1(x).astype(x_dtype))
        x = x + self.layer_scale_1(attn_update)
        ffn_hidden = self.linear1(self.norm2(x).astype(x_dtype)).astype(x_dtype)
        ffn_update = self.linear2(nn.gelu(ffn_hidden)).astype(x_dtype)
        return x + self.layer_scale_2(ffn_update)


class MossAudioTokenizerTransformer(nn.Module):
    """Codec transformer stack."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        *,
        dim_feedforward: int,
        causal: bool,
        context: int | None,
        positional_embedding: str,
        max_period: float,
        layer_scale: float | None,
    ):
        super().__init__()
        self.positional_embedding = positional_embedding
        self.max_period = max_period
        self.layers = [
            MossAudioTokenizerTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=dim_feedforward,
                causal=causal,
                context=context,
                positional_embedding=positional_embedding,
                max_period=max_period,
                layer_scale=layer_scale,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        if self.positional_embedding in {"sin", "sin_rope"}:
            x = x + _create_sinusoidal_embedding(
                x.shape[1],
                x.shape[2],
                max_period=self.max_period,
                dtype=x.dtype,
            )
        for layer in self.layers:
            x = layer(x)
        return x


class MossAudioTokenizerProjectedTransformer(nn.Module):
    """Transformer stage with channel-last projections around it."""

    def __init__(self, config: MossAudioTokenizerModuleConfig, *, context: int):
        super().__init__()
        params = dict(config.params)
        input_dimension = int(params["input_dimension"])
        output_dimension = int(params["output_dimension"])
        d_model = int(params["d_model"])
        self.module_type = config.module_type
        self.downsample_ratio = 1
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_proj = (
            nn.Linear(input_dimension, d_model, bias=False)
            if d_model != input_dimension
            else nn.Identity()
        )
        self.transformer = MossAudioTokenizerTransformer(
            d_model=d_model,
            num_heads=int(params["num_heads"]),
            num_layers=int(params["num_layers"]),
            dim_feedforward=int(params["dim_feedforward"]),
            causal=bool(params.get("causal", True)),
            context=context,
            positional_embedding=str(params.get("positional_embedding", "rope")),
            max_period=float(params.get("max_period", 10_000.0)),
            layer_scale=(
                float(params["layer_scale"]) if params.get("layer_scale") is not None else None
            ),
        )
        self.output_proj = (
            nn.Linear(d_model, output_dimension, bias=False)
            if d_model != output_dimension
            else nn.Identity()
        )

    def __call__(self, x: mx.array, input_lengths: mx.array) -> tuple[mx.array, mx.array]:
        x = self.input_proj(x.transpose(0, 2, 1)).astype(MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE)
        x = self.transformer(x)
        x = self.output_proj(x).astype(MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE).transpose(0, 2, 1)
        return x, input_lengths


class MossAudioTokenizerPatchedPretransform(nn.Module):
    """Patch up/down-sampling used around the transformer stages."""

    def __init__(self, patch_size: int, *, is_downsample: bool):
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio = patch_size
        self.is_downsample = is_downsample

    def encode(self, x: mx.array, input_lengths: mx.array) -> tuple[mx.array, mx.array]:
        batch_size, channels, _ = x.shape
        patch_size = self.patch_size
        x = x.reshape(batch_size, channels, -1, patch_size)
        x = x.transpose(0, 1, 3, 2).reshape(batch_size, channels * patch_size, -1)
        return x, input_lengths // patch_size

    def decode(self, x: mx.array, input_lengths: mx.array) -> tuple[mx.array, mx.array]:
        batch_size, channels_times_patch, length = x.shape
        patch_size = self.patch_size
        channels = channels_times_patch // patch_size
        x = x.reshape(batch_size, channels, patch_size, length)
        x = x.transpose(0, 1, 3, 2).reshape(batch_size, channels, length * patch_size)
        return x, input_lengths * patch_size

    def __call__(self, x: mx.array, input_lengths: mx.array) -> tuple[mx.array, mx.array]:
        if self.is_downsample:
            return self.encode(x, input_lengths)
        return self.decode(x, input_lengths)


class MossAudioTokenizerLFQ(nn.Module):
    """One LFQ codebook layer for codec encode/decode."""

    def __init__(self, config: MossAudioTokenizerQuantizerConfig):
        super().__init__()
        input_dim = config.rvq_dim
        codebook_dim = config.codebook_dim
        self.codebook_size = config.codebook_size
        if input_dim != codebook_dim:
            self.in_proj = MossAudioTokenizerPointwiseConv1d(input_dim, codebook_dim, bias=True)
            self.out_proj = MossAudioTokenizerPointwiseConv1d(codebook_dim, input_dim, bias=True)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        self.codebook = nn.Embedding(config.codebook_size, codebook_dim)

    def decode_code_wo_out_proj(self, embed_id: mx.array) -> mx.array:
        return self.codebook(embed_id).transpose(0, 2, 1)

    def decode_code(self, embed_id: mx.array) -> mx.array:
        return self.out_proj(self.decode_code_wo_out_proj(embed_id))

    def decode_latents(self, latents: mx.array) -> tuple[mx.array, mx.array]:
        encodings = latents.transpose(0, 2, 1).reshape(-1, latents.shape[1]).astype(mx.float32)
        codebook_ids = mx.arange(self.codebook_size, dtype=mx.int32)
        codebook = self.codebook(codebook_ids).astype(mx.float32)

        enc_norm = mx.sqrt(mx.sum(encodings * encodings, axis=1, keepdims=True) + 1e-12)
        code_norm = mx.sqrt(mx.sum(codebook * codebook, axis=1, keepdims=True) + 1e-12)
        encodings = encodings / enc_norm
        codebook = codebook / code_norm

        dist = (
            mx.sum(encodings * encodings, axis=1, keepdims=True)
            - 2.0 * (encodings @ codebook.T)
            + mx.sum(codebook * codebook, axis=1, keepdims=True).T
        )
        indices = mx.argmax(-dist, axis=1).astype(mx.int32)
        indices = indices.reshape(latents.shape[0], -1)
        z_q = self.decode_code_wo_out_proj(indices).astype(mx.float32)
        return z_q, indices

    def __call__(self, z: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        z = z.astype(mx.float32)
        z_e = self.in_proj(z).astype(mx.float32)
        z_q, indices = self.decode_latents(z_e)
        z_q = self.out_proj(z_q).astype(mx.float32)
        return z_q, indices, z_e


class MossAudioTokenizerResidualLFQ(nn.Module):
    """Residual LFQ encode/decode path."""

    def __init__(self, config: MossAudioTokenizerQuantizerConfig):
        super().__init__()
        self.input_dim = config.input_dim
        self.rvq_dim = config.rvq_dim
        self.output_dim = config.output_dim
        self.num_quantizers = config.num_quantizers
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        self.input_proj = (
            MossAudioTokenizerPointwiseConv1d(config.input_dim, config.rvq_dim, bias=True)
            if config.input_dim != config.rvq_dim
            else nn.Identity()
        )
        self.output_proj = (
            MossAudioTokenizerPointwiseConv1d(config.rvq_dim, config.output_dim, bias=True)
            if config.rvq_dim != config.output_dim
            else nn.Identity()
        )
        self.quantizers = [MossAudioTokenizerLFQ(config) for _ in range(config.num_quantizers)]

    def decode_codes(self, codes: mx.array) -> mx.array:
        num_quantizers, batch_size, frames = codes.shape
        embeddings = mx.zeros((batch_size, self.rvq_dim, frames), dtype=mx.float32)
        for quantizer_index, quantizer in enumerate(self.quantizers[:num_quantizers]):
            embeddings = embeddings + quantizer.decode_code(codes[quantizer_index]).astype(mx.float32)
        return self.output_proj(embeddings).astype(MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE)

    def __call__(
        self,
        z: mx.array,
        input_length: mx.array,
        n_quantizers: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array]:
        z = self.input_proj(z).astype(mx.float32)

        batch_size, _, max_time = z.shape
        mask = mx.arange(max_time, dtype=mx.int32)[None, :] < input_length[:, None]

        quantized_out = mx.zeros(z.shape, dtype=mx.float32)
        residual = z.astype(mx.float32)
        all_indices: list[mx.array] = []
        num_quantizers = n_quantizers or self.num_quantizers

        for quantizer_index, quantizer in enumerate(self.quantizers):
            if quantizer_index >= num_quantizers:
                break
            masked_residual = residual * mask[:, None, :]
            z_q_i, indices_i, _ = quantizer(masked_residual)
            update_mask = mask[:, None, :]
            quantized_out = quantized_out + (z_q_i * update_mask)
            residual = residual - (z_q_i * update_mask)
            all_indices.append(indices_i.astype(mx.int32))

        stacked_indices = (
            mx.stack(all_indices, axis=0)
            if all_indices
            else mx.zeros((0, batch_size, max_time), dtype=mx.int32)
        )
        quantized_out = self.output_proj(quantized_out).astype(mx.float32)
        return quantized_out, stacked_indices, input_length


@dataclass(frozen=True)
class MossAudioTokenizerEncoderOutput:
    audio_codes: mx.array
    audio_codes_lengths: mx.array
    encoder_hidden_states: mx.array | None = None


@dataclass(frozen=True)
class MossAudioTokenizerDecodeOutput:
    audio: mx.array
    audio_lengths: mx.array


class MossAudioTokenizerModel(nn.Module):
    """MLX implementation of the Cat audio tokenizer."""

    def __init__(self, config: MossAudioTokenizerConfig):
        super().__init__()
        if config.quantizer_kwargs is None:
            raise ValueError("MossAudioTokenizerConfig requires quantizer_kwargs.")
        if config.quantizer_type not in {"rlfq", "random_prefix_rlfq"}:
            raise ValueError(
                f"Only LFQ-based decode is supported for v0, got {config.quantizer_type}."
            )
        self.config = config
        self.sampling_rate = config.sampling_rate
        self.downsample_rate = config.downsample_rate
        self.causal_transformer_context_duration = config.causal_transformer_context_duration
        self.quantizer = MossAudioTokenizerResidualLFQ(config.quantizer_kwargs)

        current_frame_rate = float(config.sampling_rate)
        self.encoder = []
        for stage in config.encoder_kwargs:
            if stage.module_type == "PatchedPretransform":
                module = MossAudioTokenizerPatchedPretransform(
                    patch_size=int(stage.params["patch_size"]),
                    is_downsample=True,
                )
            elif stage.module_type == "Transformer":
                module = MossAudioTokenizerProjectedTransformer(
                    stage,
                    context=int(current_frame_rate * config.causal_transformer_context_duration),
                )
            else:
                raise ValueError(f"Unsupported encoder module_type: {stage.module_type}")
            self.encoder.append(module)
            current_frame_rate /= module.downsample_ratio

        current_frame_rate = config.frame_rate
        self.decoder = []
        for stage in config.decoder_kwargs:
            if stage.module_type == "PatchedPretransform":
                module = MossAudioTokenizerPatchedPretransform(
                    patch_size=int(stage.params["patch_size"]),
                    is_downsample=False,
                )
            elif stage.module_type == "Transformer":
                module = MossAudioTokenizerProjectedTransformer(
                    stage,
                    context=int(current_frame_rate * config.causal_transformer_context_duration),
                )
            else:
                raise ValueError(f"Unsupported decoder module_type: {stage.module_type}")
            self.decoder.append(module)
            current_frame_rate *= module.downsample_ratio

    def _encode_frame(
        self,
        input_values: mx.array,
        input_lengths: mx.array | None = None,
        n_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if input_values.ndim == 2:
            input_values = input_values[:, None, :]
        if input_values.ndim != 3:
            raise ValueError(
                "Expected input_values with shape (batch, channels, samples) "
                f"or (batch, samples), got {input_values.shape}."
            )

        batch_size, _, samples = input_values.shape
        if input_lengths is None:
            input_lengths = mx.full((batch_size,), samples, dtype=mx.int32)
        else:
            input_lengths = input_lengths.astype(mx.int32)

        if samples % self.downsample_rate != 0:
            pad_length = self.downsample_rate - (samples % self.downsample_rate)
            input_values = mx.pad(input_values, [(0, 0), (0, 0), (0, pad_length)])

        encoded = input_values.astype(MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE)
        encoded_lengths = input_lengths
        for module in self.encoder:
            encoded, encoded_lengths = module(encoded, encoded_lengths)

        _, audio_codes, audio_code_lengths = self.quantizer(
            encoded,
            encoded_lengths,
            n_quantizers,
        )
        return MossAudioTokenizerEncoderOutput(
            audio_codes=audio_codes.astype(mx.int32),
            audio_codes_lengths=audio_code_lengths.astype(mx.int32),
            encoder_hidden_states=encoded,
        )

    def _decode_frame(
        self,
        audio_codes: mx.array,
        audio_code_lengths: mx.array | None = None,
    ) -> MossAudioTokenizerDecodeOutput:
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[:, None, :]
        if audio_codes.ndim != 3:
            raise ValueError(
                "Expected audio_codes with shape (num_quantizers, batch, frames) "
                f"or (num_quantizers, frames), got {audio_codes.shape}."
            )
        _, batch_size, frames = audio_codes.shape
        if audio_code_lengths is None:
            audio_code_lengths = mx.full((batch_size,), frames, dtype=mx.int32)

        decoded = self.quantizer.decode_codes(audio_codes)
        decoded_lengths = audio_code_lengths
        for module in self.decoder:
            decoded, decoded_lengths = module(decoded, decoded_lengths)

        return MossAudioTokenizerDecodeOutput(
            audio=decoded.astype(mx.float32),
            audio_lengths=decoded_lengths,
        )

    def encode(
        self,
        input_values: mx.array,
        *,
        padding_mask: mx.array | None = None,
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if input_values.ndim == 1:
            input_values = input_values[None, :]
        if input_values.ndim == 2:
            input_values = input_values[:, None, :]

        if padding_mask is not None:
            input_lengths = mx.sum(padding_mask.astype(mx.int32), axis=-1)
        else:
            input_lengths = None
        return self._encode_frame(
            input_values.astype(MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE),
            input_lengths,
            n_quantizers=num_quantizers,
        )

    def batch_encode(
        self,
        wav_list: list[mx.array],
        *,
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerEncoderOutput:
        if not wav_list:
            raise ValueError("wav_list must contain at least one waveform.")
        max_length = max(int(wav.shape[-1]) for wav in wav_list)
        batch_size = len(wav_list)
        input_values = mx.zeros(
            (batch_size, 1, max_length),
            dtype=MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE,
        )
        input_lengths = mx.zeros((batch_size,), dtype=mx.int32)
        for batch_index, wav in enumerate(wav_list):
            waveform = wav.astype(MOSS_AUDIO_TOKENIZER_ACTIVATION_DTYPE)
            if waveform.ndim != 1:
                raise ValueError(
                    f"Expected mono waveform with shape (samples,), got {waveform.shape}."
                )
            length = int(waveform.shape[-1])
            input_values[batch_index, 0, :length] = waveform
            input_lengths[batch_index] = length
        return self._encode_frame(
            input_values,
            input_lengths,
            n_quantizers=num_quantizers,
        )

    def decode(
        self,
        audio_codes: mx.array,
        *,
        padding_mask: mx.array | None = None,
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerDecodeOutput:
        if audio_codes.ndim == 2:
            audio_codes = audio_codes[:, None, :]
        if num_quantizers is not None:
            audio_codes = audio_codes[:num_quantizers]

        if padding_mask is not None:
            code_lengths = mx.sum(padding_mask.astype(mx.int32), axis=-1)
        else:
            code_lengths = None
        return self._decode_frame(audio_codes, code_lengths)

    def batch_decode(
        self,
        codes_list: list[mx.array],
        *,
        num_quantizers: int | None = None,
    ) -> MossAudioTokenizerDecodeOutput:
        if not codes_list:
            raise ValueError("codes_list must contain at least one code tensor.")
        batch_size = len(codes_list)
        if num_quantizers is None:
            num_quantizers = int(codes_list[0].shape[0])
        max_frames = max(int(codes.shape[-1]) for codes in codes_list)
        audio_codes = mx.zeros((num_quantizers, batch_size, max_frames), dtype=mx.int32)
        audio_code_lengths = mx.zeros((batch_size,), dtype=mx.int32)
        for batch_index, codes in enumerate(codes_list):
            codes = codes[:num_quantizers]
            frames = int(codes.shape[-1])
            audio_codes[:, batch_index, :frames] = codes
            audio_code_lengths[batch_index] = frames
        return self._decode_frame(audio_codes, audio_code_lengths)
