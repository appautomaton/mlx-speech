"""FastConformer encoder assembly for Nemotron 3.5 ASR."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .attention import (
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
    create_chunked_limited_mask,
)
from .config import ConformerArgs
from .subsampling import CausalDwStridingSubsampling


class FeedForward(nn.Module):
    """Conformer position-wise feed-forward module."""

    def __init__(self, d_model: int, d_ff: int, *, use_bias: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.activation = nn.SiLU()
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.activation(self.linear1(x)))


class ConformerConvolution(nn.Module):
    """Causal depthwise convolution with NeMo-compatible parameter names."""

    def __init__(self, args: ConformerArgs) -> None:
        super().__init__()
        if args.conv_norm_type != "layer_norm":
            raise ValueError("Nemotron requires conv_norm_type='layer_norm'")

        context = args.conv_context_size
        if context == "causal":
            pad_left, pad_right = args.conv_kernel_size - 1, 0
        elif isinstance(context, tuple) and len(context) == 2:
            pad_left, pad_right = context
        else:
            raise ValueError("conv_context_size must be 'causal' or a (left, right) pair")
        if pad_left + pad_right + 1 != args.conv_kernel_size:
            raise ValueError("conv context must span conv_kernel_size frames")

        self.kernel_size = args.conv_kernel_size
        self.pad_left = int(pad_left)
        self.pad_right = int(pad_right)
        self.pointwise_conv1 = nn.Conv1d(
            args.d_model,
            args.d_model * 2,
            kernel_size=1,
            bias=args.use_bias,
        )
        self.depthwise_conv = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=args.conv_kernel_size,
            groups=args.d_model,
            bias=args.use_bias,
        )
        # NeMo retains this attribute name when norm_type is layer_norm.
        self.batch_norm = nn.LayerNorm(args.d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            args.d_model,
            args.d_model,
            kernel_size=1,
            bias=args.use_bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.glu(self.pointwise_conv1(x), axis=-1)
        x = mx.pad(x, ((0, 0), (self.pad_left, self.pad_right), (0, 0)))
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.pointwise_conv2(x)


class ConformerBlock(nn.Module):
    """Macaron Conformer block matching the NeMo checkpoint hierarchy."""

    def __init__(self, args: ConformerArgs) -> None:
        super().__init__()
        d_ff = args.d_model * args.ff_expansion_factor
        self.norm_feed_forward1 = nn.LayerNorm(args.d_model)
        self.feed_forward1 = FeedForward(
            args.d_model, d_ff, use_bias=args.use_bias
        )
        self.norm_self_att = nn.LayerNorm(args.d_model)
        self.self_attn = RelPositionMultiHeadAttention(
            args.n_heads, args.d_model, use_bias=args.use_bias
        )
        self.norm_conv = nn.LayerNorm(args.d_model)
        self.conv = ConformerConvolution(args)
        self.norm_feed_forward2 = nn.LayerNorm(args.d_model)
        self.feed_forward2 = FeedForward(
            args.d_model, d_ff, use_bias=args.use_bias
        )
        self.norm_out = nn.LayerNorm(args.d_model)

    def __call__(
        self,
        x: mx.array,
        pos_emb: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class FastConformerEncoder(nn.Module):
    """Nemotron's causal 8x subsampler followed by FastConformer blocks."""

    def __init__(self, args: ConformerArgs | None = None) -> None:
        super().__init__()
        self.args = args or ConformerArgs()
        if not self.args.causal_downsampling:
            raise ValueError("Nemotron requires causal_downsampling=True")
        if self.args.self_attention_model != "rel_pos":
            raise ValueError("Nemotron requires self_attention_model='rel_pos'")
        if self.args.att_context_style != "chunked_limited":
            raise ValueError("Nemotron requires att_context_style='chunked_limited'")

        self.pre_encode = CausalDwStridingSubsampling(
            feat_in=self.args.feat_in,
            d_model=self.args.d_model,
            conv_channels=self.args.subsampling_conv_channels,
            subsampling_factor=self.args.subsampling_factor,
        )
        self.pos_enc = RelPositionalEncoding(
            self.args.d_model,
            max_len=self.args.pos_emb_max_len,
            scale_input=self.args.xscaling,
        )
        self.layers = [ConformerBlock(self.args) for _ in range(self.args.n_layers)]

    def __call__(
        self,
        features: mx.array,
        lengths: mx.array | None = None,
        att_context_size: tuple[int, int] | None = None,
    ) -> tuple[mx.array, mx.array]:
        if features.ndim != 3:
            raise ValueError(f"expected features [B, T, F], got {features.shape}")
        if features.shape[0] != 1:
            raise ValueError("Nemotron currently supports batch size 1 only")
        if lengths is None:
            lengths = mx.array([features.shape[1]], dtype=mx.int32)
        if lengths.shape != (1,):
            raise ValueError("lengths must have shape (1,)")

        x, out_lengths = self.pre_encode(features, lengths)
        # Padded batching is deliberately out of scope. Trim the sole example so
        # neither attention nor convolution observes invalid encoder frames.
        valid_length = int(out_lengths[0].item())
        x = x[:, :valid_length]
        x, pos_emb = self.pos_enc(x)

        context = att_context_size or self.args.default_att_context_size
        left_context, right_context = context
        mask = create_chunked_limited_mask(
            x.shape[1], int(left_context), int(right_context)
        ).astype(x.dtype)
        for layer in self.layers:
            x = layer(x, pos_emb, mask)
        return x, out_lengths


__all__ = [
    "ConformerBlock",
    "ConformerConvolution",
    "FastConformerEncoder",
    "FeedForward",
]
