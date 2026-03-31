"""ParakeetEncoder (Fast-Conformer) for CohereAsr in pure MLX."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import ParakeetEncoderConfig


# ---------------------------------------------------------------------------
# Relative positional encoding (Transformer-XL style)
# ---------------------------------------------------------------------------

class ParakeetRelPosEncoding(nn.Module):
    """Sinusoidal relative positional encoding covering 2T-1 positions."""

    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        # inv_freq is a buffer — not a learned parameter
        d = config.hidden_size
        inv_freq = 1.0 / (
            10000.0 ** (mx.arange(0, d, 2, dtype=mx.float32) / d)
        )
        self._inv_freq = inv_freq  # (d/2,)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        """Compute relative position embeddings for the given sequence."""
        batch_size, seq_length = hidden_states.shape[:2]
        if seq_length > self.max_position_embeddings:
            raise ValueError(
                f"Sequence Length: {seq_length} has to be less or equal than "
                f"config.max_position_embeddings {self.max_position_embeddings}."
            )

        position_ids = mx.arange(seq_length - 1, -seq_length, -1, dtype=mx.float32)
        inv_freq_expanded = mx.broadcast_to(
            self._inv_freq[None, :, None],
            (batch_size, self._inv_freq.shape[0], 1),
        ).astype(mx.float32)
        position_ids_expanded = position_ids[None, None, :].astype(mx.float32)

        freqs = mx.matmul(inv_freq_expanded, position_ids_expanded).transpose(0, 2, 1)
        sin = mx.sin(freqs)
        cos = mx.cos(freqs)
        pos_emb = mx.stack([sin, cos], axis=-1)
        pos_emb = pos_emb.reshape(batch_size, 2 * seq_length - 1, self.hidden_size)
        return pos_emb.astype(hidden_states.dtype)


# ---------------------------------------------------------------------------
# 2D subsampling conv stack (dw_striding, factor=8 → 3 layers of stride-2)
# ---------------------------------------------------------------------------

class ParakeetSubsampling(nn.Module):
    """Conv2d-based 8× temporal subsampling treating mel as a 2D image.

    Input:  (batch, T_mel, n_mels)          -- channels-last in MLX
    Output: (batch, T', hidden_size)

    Layer layout (dw_striding, n_layers=3):
        conv0            — standard Conv2d(1 → ch, k, stride=s)
        _dw_weight_0/1   — depthwise Conv2d (groups=ch); stored as plain arrays
                           because MLX nn.Conv2d has no groups parameter
        pw_convs[0/1]    — pointwise Conv2d(ch → ch, 1×1)
    """

    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        k = config.subsampling_conv_kernel_size  # 3
        s = config.subsampling_conv_stride       # 2
        ch = config.subsampling_conv_channels    # 256
        n_layers = config.num_subsampling_layers  # 3
        # Store for use in __call__ and output_lengths
        self._dw_k: int = k
        self._dw_s: int = s
        self._dw_ch: int = ch
        self._n_layers: int = n_layers

        # First conv: standard Conv2d(1 → ch)
        self.conv0 = nn.Conv2d(1, ch, kernel_size=k, stride=s, padding=k // 2)
        self.relu0 = nn.ReLU()

        # Depthwise Conv2d weights (groups=ch). MLX nn.Conv2d has no groups parameter;
        # we call mx.conv2d at forward time. Named without leading underscore so MLX
        # includes them in the parameter tree.
        # Shape: (C_out, H, W, C_in/groups) = (ch, k, k, 1) in MLX Conv2d layout.
        self.dw_weight_0 = mx.zeros((ch, k, k, 1))
        self.dw_bias_0 = mx.zeros((ch,))
        self.dw_weight_1 = mx.zeros((ch, k, k, 1))
        self.dw_bias_1 = mx.zeros((ch,))

        # Pointwise 1×1 convolutions
        self.pw_convs = [nn.Conv2d(ch, ch, kernel_size=1) for _ in range(n_layers - 1)]
        self.relus = [nn.ReLU() for _ in range(n_layers - 1)]

        # After 3× stride-2: freq dim = n_mels // s^n_layers = 128 // 8 = 16
        freq_out = config.num_mel_bins // (s ** n_layers)
        self.linear = nn.Linear(ch * freq_out, config.hidden_size, bias=True)

    def __call__(self, features: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """
        Args:
            features: (batch, T_mel, n_mels) float32
        Returns:
            (batch, T', hidden_size)
        """
        def apply_time_mask(hidden: mx.array, lengths: mx.array | None) -> mx.array:
            if lengths is None:
                return hidden
            current_seq_length = hidden.shape[1]
            valid = mx.arange(current_seq_length)[None, :] < lengths[:, None]
            return hidden * valid[:, :, None, None].astype(hidden.dtype)

        current_lengths: mx.array | None = None
        if attention_mask is not None:
            current_lengths = attention_mask.sum(axis=-1).astype(mx.int32)

        # Add channel dim: (batch, T, n_mels, 1)
        x = features[:, :, :, None]
        x = self.relu0(self.conv0(x))
        if current_lengths is not None:
            current_lengths = mx.floor(
                (
                    current_lengths.astype(mx.float32)
                    + (self._dw_k // 2) * 2
                    - self._dw_k
                )
                / self._dw_s
            ).astype(mx.int32) + 1
            x = apply_time_mask(x, current_lengths)

        dw_weights = [self.dw_weight_0, self.dw_weight_1]
        dw_biases = [self.dw_bias_0, self.dw_bias_1]
        for dw_w, dw_b, pw, relu in zip(dw_weights, dw_biases, self.pw_convs, self.relus):
            x = mx.conv2d(
                x, dw_w,
                stride=(self._dw_s, self._dw_s),
                padding=(self._dw_k // 2, self._dw_k // 2),
                groups=self._dw_ch,
            )
            if dw_b is not None:
                x = x + dw_b
            x = relu(pw(x))
            if current_lengths is not None:
                current_lengths = mx.floor(
                    (
                        current_lengths.astype(mx.float32)
                        + (self._dw_k // 2) * 2
                        - self._dw_k
                    )
                    / self._dw_s
                ).astype(mx.int32) + 1
                x = apply_time_mask(x, current_lengths)

        # HF flattens (channels, freq_out) per timestep after transpose(1, 2),
        # so reorder from MLX channels-last (T, freq_out, ch) to (T, ch, freq_out)
        # before the final projection.
        batch, t_out, freq_out, ch = x.shape
        x = x.transpose(0, 1, 3, 2).reshape(batch, t_out, ch * freq_out)
        x = self.linear(x)  # (batch, T', hidden_size)
        return x

    def output_lengths(self, input_lengths: mx.array) -> mx.array:
        """Compute subsampled sequence lengths from input lengths.

        For Conv2d(k=3, s=2, pad=1): out = floor((L-1)/2) + 1
        """
        lengths = input_lengths
        for _ in range(self._n_layers):
            lengths = mx.floor((lengths.astype(mx.float32) - 1) / 2).astype(mx.int32) + 1
        return lengths


# ---------------------------------------------------------------------------
# Feed-forward module
# ---------------------------------------------------------------------------

class ParakeetFeedForward(nn.Module):
    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.attention_bias)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.attention_bias)
        self.hidden_act = config.hidden_act
        self.activation_dropout = config.activation_dropout

    def __call__(self, x: mx.array) -> mx.array:
        x = self.linear1(x)
        if self.hidden_act != "silu":
            raise ValueError(f"Unsupported Parakeet hidden_act: {self.hidden_act}")
        x = nn.silu(x)
        return self.linear2(x)


# ---------------------------------------------------------------------------
# Convolution module (GLU + depthwise + BatchNorm + silu)
# ---------------------------------------------------------------------------

class ParakeetConvModule(nn.Module):
    """Conformer convolution sub-module (channels-last throughout)."""

    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        ch = config.hidden_size
        k = config.conv_kernel_size
        bias = config.convolution_bias

        # pointwise_conv1: expand to 2ch for GLU
        self.pointwise_conv1 = nn.Conv1d(ch, 2 * ch, kernel_size=1, bias=bias)
        # Depthwise conv: MLX nn.Conv1d has no groups parameter; use mx.conv1d at call time.
        # Named without leading underscore so MLX includes them in the parameter tree.
        # Shape: (C_out, L, C_in/groups) = (ch, k, 1) in MLX Conv1d layout.
        self.dw_weight = mx.zeros((ch, k, 1))
        self.dw_bias = mx.zeros((ch,)) if bias else None
        self._dw_kernel_size = k
        self._dw_padding = k // 2
        # BatchNorm parameters (stored as arrays, applied manually at inference)
        self.norm = nn.BatchNorm(ch, eps=1e-5)
        self.pointwise_conv2 = nn.Conv1d(ch, ch, kernel_size=1, bias=bias)

    def __call__(self, x: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        """
        Args:
            x: (batch, time, channels) channels-last
        Returns:
            (batch, time, channels)
        """
        # Expand and GLU: value = first half, gate = second half (matches nn.functional.glu dim=1)
        x = self.pointwise_conv1(x)  # (batch, time, 2*ch)
        half = x.shape[-1] // 2
        x = x[..., :half] * mx.sigmoid(x[..., half:])  # (batch, time, ch)

        if attention_mask is not None:
            if attention_mask.dtype == mx.bool_:
                valid_rows = ~mx.all(~attention_mask, axis=2)
            else:
                valid_rows = ~mx.all(attention_mask != 0.0, axis=2)
            x = x * valid_rows[:, 0, :, None].astype(x.dtype)

        # Depthwise conv: mx.conv1d with groups=ch (channels-last, ch = x.shape[-1])
        x = mx.conv1d(x, self.dw_weight, stride=1, padding=self._dw_padding, groups=x.shape[-1])
        if self.dw_bias is not None:
            x = x + self.dw_bias

        # BatchNorm (inference: use running stats)
        x = self.norm(x)

        x = nn.silu(x)
        x = self.pointwise_conv2(x)  # (batch, time, ch)
        return x


# ---------------------------------------------------------------------------
# Multi-head attention with Transformer-XL relative positional encoding
# ---------------------------------------------------------------------------

class ParakeetAttention(nn.Module):
    """Bidirectional self-attention with relative positional encoding."""

    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        bias = config.attention_bias

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        # W_{k,R}: positional key projection (no bias)
        self.relative_k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        # Global learnable biases
        self.bias_u = mx.zeros((config.num_attention_heads, config.head_dim))
        self.bias_v = mx.zeros((config.num_attention_heads, config.head_dim))

    def _rel_shift(self, scores: mx.array) -> mx.array:
        """Relative shift trick (appendix B of 1901.02860).

        Args:
            scores: (batch, heads, query_len, 2*query_len - 1)
        Returns:
            (batch, heads, query_len, 2*query_len - 1)  after shift
        """
        batch, heads, q_len, pos_len = scores.shape
        # Pad one zero on the left of the position axis
        scores = mx.pad(scores, [(0, 0), (0, 0), (0, 0), (1, 0)])
        # Reshape and slice to shift
        scores = scores.reshape(batch, heads, pos_len + 1, q_len)
        scores = scores[:, :, 1:]  # remove first row
        scores = scores.reshape(batch, heads, q_len, pos_len)
        return scores

    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, seq_length = hidden_states.shape[:2]
        hidden_shape = (batch_size, seq_length, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)
        value_states = self.v_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)

        query_states_with_bias_u = query_states + self.bias_u.reshape(1, self.num_heads, 1, self.head_dim)
        query_states_with_bias_v = query_states + self.bias_v.reshape(1, self.num_heads, 1, self.head_dim)

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.reshape(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim,
        )

        matrix_bd = query_states_with_bias_v @ relative_key_states.transpose(0, 2, 3, 1)
        matrix_bd = self._rel_shift(matrix_bd)
        matrix_bd = matrix_bd[..., :seq_length]
        matrix_bd = matrix_bd * self.scale

        if attention_mask is not None:
            neg_inf = mx.array(mx.finfo(mx.float32).min, dtype=matrix_bd.dtype)
            if attention_mask.dtype == mx.bool_:
                matrix_bd = mx.where(attention_mask, matrix_bd, neg_inf)
            else:
                matrix_bd = mx.where(attention_mask == 0.0, matrix_bd, neg_inf)

        attention_scores = (query_states_with_bias_u @ key_states.transpose(0, 1, 3, 2)) * self.scale
        attention_scores = attention_scores + matrix_bd

        weights = mx.softmax(attention_scores.astype(mx.float32), axis=-1).astype(hidden_states.dtype)
        out = (weights @ value_states).transpose(0, 2, 1, 3).reshape(batch_size, seq_length, -1)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# Conformer block
# ---------------------------------------------------------------------------

class ParakeetConformerBlock(nn.Module):
    """Single Fast-Conformer block: FF(0.5) → attn → conv → FF(0.5) → LN."""

    def __init__(self, config: ParakeetEncoderConfig, layer_idx: int):
        super().__init__()
        self.feed_forward1 = ParakeetFeedForward(config)
        self.self_attn = ParakeetAttention(config, layer_idx)
        self.conv = ParakeetConvModule(config)
        self.feed_forward2 = ParakeetFeedForward(config)

        self.norm_feed_forward1 = nn.LayerNorm(config.hidden_size)
        self.norm_self_att = nn.LayerNorm(config.hidden_size)
        self.norm_conv = nn.LayerNorm(config.hidden_size)
        self.norm_feed_forward2 = nn.LayerNorm(config.hidden_size)
        self.norm_out = nn.LayerNorm(config.hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: mx.array | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * hidden_states

        normalized_hidden_states = self.norm_self_att(hidden_states)
        hidden_states = hidden_states + self.self_attn(
            normalized_hidden_states,
            position_embeddings,
            attention_mask,
        )

        hidden_states = hidden_states + self.conv(
            self.norm_conv(hidden_states),
            attention_mask=attention_mask,
        )

        hidden_states = hidden_states + 0.5 * self.feed_forward2(self.norm_feed_forward2(hidden_states))
        hidden_states = self.norm_out(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------

class ParakeetEncoder(nn.Module):
    """Fast-Conformer encoder: subsampling → 48 × ConformerBlock."""

    def __init__(self, config: ParakeetEncoderConfig):
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.dropout_positions = config.dropout_positions
        self.layerdrop = config.layerdrop
        self.input_scale = math.sqrt(config.hidden_size) if config.scale_input else 1.0

        self.subsampling = ParakeetSubsampling(config)
        self.pos_encoding = ParakeetRelPosEncoding(config)
        self.layers = [
            ParakeetConformerBlock(config, i)
            for i in range(config.num_hidden_layers)
        ]

    def __call__(
        self,
        features: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array | None]:
        """
        Args:
            features:       (batch, T_mel, n_mels)
            attention_mask: (batch, T_mel) bool or None

        Returns:
            hidden_states:       (batch, T', hidden_size)
            output_attention_mask: (batch, T') bool or None
        """
        hidden_states = self.subsampling(features, attention_mask)
        hidden_states = hidden_states * self.input_scale
        position_embeddings = self.pos_encoding(hidden_states)

        output_mask: mx.array | None = None
        encoder_attention_mask: mx.array | None = None
        if attention_mask is not None:
            input_lengths = attention_mask.sum(axis=-1).astype(mx.int32)
            output_lengths = self.subsampling.output_lengths(input_lengths)
            target_length = hidden_states.shape[1]
            output_mask = mx.arange(target_length)[None, :] < output_lengths[:, None]
            encoder_attention_mask = mx.broadcast_to(
                output_mask[:, None, :],
                (output_mask.shape[0], target_length, target_length),
            )
            encoder_attention_mask = encoder_attention_mask & encoder_attention_mask.transpose(0, 2, 1)
            encoder_attention_mask = encoder_attention_mask[:, None, :, :]

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=encoder_attention_mask,
                position_embeddings=position_embeddings,
            )

        return hidden_states, output_mask
