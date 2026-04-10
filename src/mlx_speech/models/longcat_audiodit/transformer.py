"""MLX-native AudioDiT transformer backbone."""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn

from .config import LongCatAudioDiTConfig


class _Identity(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return x


class _SiLU(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.silu(x)


class _GeluTanh(nn.Module):
    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu(x)


class _LayerNormNoAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x.astype(mx.float32), axis=-1, keepdims=True)
        variance = mx.mean(
            mx.square(x.astype(mx.float32) - mean), axis=-1, keepdims=True
        )
        return ((x.astype(mx.float32) - mean) * mx.rsqrt(variance + self.eps)).astype(
            x.dtype
        )


class AudioDiTRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        variance = mx.mean(mx.square(x.astype(mx.float32)), axis=-1, keepdims=True)
        return x * mx.rsqrt(variance + self.eps) * self.weight


class AudioDiTSinusPositionEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def __call__(self, x: mx.array, scale: float = 1000.0) -> mx.array:
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = mx.exp(mx.arange(half_dim, dtype=mx.float32) * -emb)
        emb = scale * x[:, None] * emb[None, :]
        return mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)


class AudioDiTTimestepEmbedding(nn.Module):
    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.time_embed = AudioDiTSinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = [
            nn.Linear(freq_embed_dim, dim, bias=True),
            _SiLU(),
            nn.Linear(dim, dim, bias=True),
        ]

    def __call__(self, timestep: mx.array) -> mx.array:
        hidden = self.time_embed(timestep)
        for layer in self.time_mlp:
            hidden = layer(hidden)
        return hidden


class AudioDiTRotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 100000.0
    ) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    def __call__(self, seq_len: int, dtype) -> tuple[mx.array, mx.array]:
        length = max(seq_len, self.max_position_embeddings)
        inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.float32) / self.dim)
        )
        t = mx.arange(length, dtype=mx.float32)
        freqs = t[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype)[:seq_len], mx.sin(emb).astype(dtype)[:seq_len]


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_emb(x: mx.array, rope: tuple[mx.array, mx.array]) -> mx.array:
    cos, sin = rope
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return (
        x.astype(mx.float32) * cos + _rotate_half(x).astype(mx.float32) * sin
    ).astype(x.dtype)


class AudioDiTGRN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = mx.zeros((1, 1, dim), dtype=mx.float32)
        self.beta = mx.zeros((1, 1, dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        gx = mx.sqrt(mx.sum(mx.square(x), axis=1, keepdims=True))
        nx = gx / (mx.mean(gx, axis=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class AudioDiTConvNeXtV2Block(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        *,
        dilation: int = 1,
        kernel_size: int = 7,
        bias: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,
            dilation=dilation,
            bias=bias,
        )
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pwconv1 = nn.Linear(dim, intermediate_dim, bias=bias)
        self.act = _SiLU()
        self.grn = AudioDiTGRN(intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class AudioDiTEmbedder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = [
            nn.Linear(in_dim, out_dim, bias=True),
            _SiLU(),
            nn.Linear(out_dim, out_dim, bias=True),
        ]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        if mask is not None:
            x = mx.where(mask[..., None], x, mx.zeros_like(x))
        for layer in self.proj:
            x = layer(x)
        if mask is not None:
            x = mx.where(mask[..., None], x, mx.zeros_like(x))
        return x


class AudioDiTAdaLNMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = True) -> None:
        super().__init__()
        self.mlp = [
            _SiLU(),
            nn.Linear(in_dim, out_dim, bias=bias),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.mlp:
            x = layer(x)
        return x


class AudioDiTAdaLayerNormZeroFinal(nn.Module):
    def __init__(self, dim: int, *, bias: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.silu = _SiLU()
        self.linear = nn.Linear(dim, dim * 2, bias=bias)
        self.norm = _LayerNormNoAffine(dim, eps=eps)

    def __call__(self, x: mx.array, emb: mx.array) -> mx.array:
        emb = self.linear(self.silu(emb))
        scale, shift = mx.split(emb, 2, axis=-1)
        x = self.norm(x)
        return x * (1 + scale[:, None, :]) + shift[:, None, :]


def _modulate(
    x: mx.array, scale: mx.array, shift: mx.array, *, eps: float = 1e-6
) -> mx.array:
    normed = _LayerNormNoAffine(x.shape[-1], eps=eps)(x)
    return normed * (1 + scale[:, None, :]) + shift[:, None, :]


class AudioDiTSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        del dropout
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
            self.k_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
        self.to_out = [nn.Linear(self.inner_dim, dim, bias=bias), _Identity()]

    def __call__(
        self,
        x: mx.array,
        *,
        mask: mx.array | None = None,
        rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        head_dim = self.inner_dim // self.heads
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = mx.transpose(
            query.reshape(batch, seq_len, self.heads, head_dim), (0, 2, 1, 3)
        )
        key = mx.transpose(
            key.reshape(batch, seq_len, self.heads, head_dim), (0, 2, 1, 3)
        )
        value = mx.transpose(
            value.reshape(batch, seq_len, self.heads, head_dim), (0, 2, 1, 3)
        )
        if rope is not None:
            query = _apply_rotary_emb(query, rope)
            key = _apply_rotary_emb(key, rope)
        scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2))) / math.sqrt(head_dim)
        if mask is not None:
            attn_mask = mask[:, None, None, :]
            scores = mx.where(
                attn_mask, scores, mx.full(scores.shape, -1e9, dtype=scores.dtype)
            )
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = mx.matmul(probs, value)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, seq_len, self.inner_dim)
        for layer in self.to_out:
            out = layer(out)
        return out


class AudioDiTCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        q_dim: int,
        kv_dim: int,
        heads: int,
        dim_head: int,
        dropout: float = 0.0,
        bias: bool = True,
        qk_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        del dropout
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(q_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.to_v = nn.Linear(kv_dim, self.inner_dim, bias=bias)
        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
            self.k_norm = AudioDiTRMSNorm(self.inner_dim, eps=eps)
        self.to_out = [nn.Linear(self.inner_dim, q_dim, bias=bias), _Identity()]

    def __call__(
        self,
        *,
        x: mx.array,
        cond: mx.array,
        mask: mx.array | None = None,
        cond_mask: mx.array | None = None,
        rope: tuple[mx.array, mx.array] | None = None,
        cond_rope: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch, seq_len, _ = x.shape
        cond_seq_len = cond.shape[1]
        head_dim = self.inner_dim // self.heads
        query = self.to_q(x)
        key = self.to_k(cond)
        value = self.to_v(cond)
        if self.qk_norm:
            query = self.q_norm(query)
            key = self.k_norm(key)
        query = mx.transpose(
            query.reshape(batch, seq_len, self.heads, head_dim), (0, 2, 1, 3)
        )
        key = mx.transpose(
            key.reshape(batch, cond_seq_len, self.heads, head_dim), (0, 2, 1, 3)
        )
        value = mx.transpose(
            value.reshape(batch, cond_seq_len, self.heads, head_dim), (0, 2, 1, 3)
        )
        if rope is not None:
            query = _apply_rotary_emb(query, rope)
        if cond_rope is not None:
            key = _apply_rotary_emb(key, cond_rope)
        scores = mx.matmul(query, mx.transpose(key, (0, 1, 3, 2))) / math.sqrt(head_dim)
        if mask is not None and cond_mask is not None:
            attn_mask = cond_mask[:, None, None, :]
            scores = mx.where(
                attn_mask, scores, mx.full(scores.shape, -1e9, dtype=scores.dtype)
            )
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        out = mx.matmul(probs, value)
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(batch, seq_len, self.inner_dim)
        for layer in self.to_out:
            out = layer(out)
        return out


class AudioDiTFeedForward(nn.Module):
    def __init__(
        self, *, dim: int, mult: float = 4.0, dropout: float = 0.0, bias: bool = True
    ) -> None:
        super().__init__()
        del dropout
        inner_dim = int(dim * mult)
        self.ff = [
            nn.Linear(dim, inner_dim, bias=bias),
            _GeluTanh(),
            _Identity(),
            nn.Linear(inner_dim, dim, bias=bias),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.ff:
            x = layer(x)
        return x


class AudioDiTBlock(nn.Module):
    def __init__(self, config: LongCatAudioDiTConfig) -> None:
        super().__init__()
        dim = config.dit_dim
        heads = config.dit_heads
        dim_head = dim // heads
        self.adaln_type = config.dit_adaln_type
        self.adaln_use_text_cond = config.dit_adaln_use_text_cond
        if self.adaln_type == "local":
            self.adaln_mlp = AudioDiTAdaLNMLP(dim, dim * 6, bias=True)
        else:
            self.adaln_scale_shift = mx.random.normal(
                (dim * 6,), dtype=mx.float32
            ) / math.sqrt(dim)

        self.self_attn = AudioDiTSelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=config.dit_dropout,
            bias=config.dit_bias,
            qk_norm=config.dit_qk_norm,
            eps=config.dit_eps,
        )
        self.use_cross_attn = config.dit_cross_attn
        if self.use_cross_attn:
            self.cross_attn = AudioDiTCrossAttention(
                q_dim=dim,
                kv_dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=config.dit_dropout,
                bias=config.dit_bias,
                qk_norm=config.dit_qk_norm,
                eps=config.dit_eps,
            )
            self.cross_attn_norm = (
                nn.LayerNorm(dim, eps=config.dit_eps)
                if config.dit_cross_attn_norm
                else _Identity()
            )
            self.cross_attn_norm_c = (
                nn.LayerNorm(dim, eps=config.dit_eps)
                if config.dit_cross_attn_norm
                else _Identity()
            )
        self.ffn = AudioDiTFeedForward(
            dim=dim,
            mult=config.dit_ff_mult,
            dropout=config.dit_dropout,
            bias=config.dit_bias,
        )
        self.eps = config.dit_eps

    def __call__(
        self,
        *,
        x: mx.array,
        t: mx.array,
        cond: mx.array,
        mask: mx.array | None = None,
        cond_mask: mx.array | None = None,
        rope: tuple[mx.array, mx.array] | None = None,
        cond_rope: tuple[mx.array, mx.array] | None = None,
        adaln_global_out: mx.array | None = None,
    ) -> mx.array:
        if self.adaln_type == "local" and adaln_global_out is None:
            if self.adaln_use_text_cond and cond_mask is not None:
                cond_mean = mx.sum(cond, axis=1) / mx.sum(
                    cond_mask.astype(cond.dtype), axis=1, keepdims=True
                )
                norm_cond = t + cond_mean
            else:
                norm_cond = t
            adaln_out = self.adaln_mlp(norm_cond)
        else:
            adaln_out = adaln_global_out + self.adaln_scale_shift[None, :]

        gate_sa, scale_sa, shift_sa, gate_ffn, scale_ffn, shift_ffn = mx.split(
            adaln_out, 6, axis=-1
        )
        norm = _modulate(x, scale_sa, shift_sa, eps=self.eps)
        attn_output = self.self_attn(norm, mask=mask, rope=rope)
        x = x + gate_sa[:, None, :] * attn_output

        if self.use_cross_attn:
            cross_out = self.cross_attn(
                x=self.cross_attn_norm(x),
                cond=self.cross_attn_norm_c(cond),
                mask=mask,
                cond_mask=cond_mask,
                rope=rope,
                cond_rope=cond_rope,
            )
            x = x + cross_out

        norm = _modulate(x, scale_ffn, shift_ffn, eps=self.eps)
        ff_output = self.ffn(norm)
        x = x + gate_ffn[:, None, :] * ff_output
        return x


class LongCatAudioDiTTransformer(nn.Module):
    def __init__(self, config: LongCatAudioDiTConfig) -> None:
        super().__init__()
        dim = config.dit_dim
        latent_dim = config.latent_dim
        dim_head = dim // config.dit_heads
        self.config = config
        self.long_skip = config.dit_long_skip
        self.adaln_type = config.dit_adaln_type
        self.adaln_use_text_cond = config.dit_adaln_use_text_cond
        self.time_embed = AudioDiTTimestepEmbedding(dim)
        self.input_embed = AudioDiTEmbedder(latent_dim, dim)
        self.text_embed = AudioDiTEmbedder(config.dit_text_dim, dim)
        self.rotary_embed = AudioDiTRotaryEmbedding(dim_head, 2048, base=100000.0)
        self.blocks = [AudioDiTBlock(config) for _ in range(config.dit_depth)]
        self.norm_out = AudioDiTAdaLayerNormZeroFinal(
            dim, bias=True, eps=config.dit_eps
        )
        self.proj_out = nn.Linear(dim, latent_dim, bias=True)
        if self.adaln_type == "global":
            self.adaln_global_mlp = AudioDiTAdaLNMLP(dim, dim * 6, bias=True)
        self.text_conv = config.dit_text_conv
        if self.text_conv:
            self.text_conv_layer = [
                AudioDiTConvNeXtV2Block(
                    dim, dim * 2, bias=config.dit_bias, eps=config.dit_eps
                )
                for _ in range(4)
            ]
        self.use_latent_condition = config.dit_use_latent_condition
        if self.use_latent_condition:
            self.latent_embed = AudioDiTEmbedder(latent_dim, dim)
            self.latent_cond_embedder = AudioDiTEmbedder(dim * 2, dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        zero = 0.0
        if self.adaln_type == "global":
            self.adaln_global_mlp.mlp[1].weight = mx.zeros_like(
                self.adaln_global_mlp.mlp[1].weight
            )
            if "bias" in self.adaln_global_mlp.mlp[1]:
                self.adaln_global_mlp.mlp[1].bias = mx.full(
                    self.adaln_global_mlp.mlp[1].bias.shape,
                    zero,
                    dtype=self.adaln_global_mlp.mlp[1].bias.dtype,
                )
        self.norm_out.linear.weight = mx.zeros_like(self.norm_out.linear.weight)
        self.proj_out.weight = mx.zeros_like(self.proj_out.weight)
        if "bias" in self.norm_out.linear:
            self.norm_out.linear.bias = mx.zeros_like(self.norm_out.linear.bias)
        if "bias" in self.proj_out:
            self.proj_out.bias = mx.zeros_like(self.proj_out.bias)

    def __call__(
        self,
        *,
        x: mx.array,
        text: mx.array,
        text_len: mx.array,
        time: mx.array,
        mask: mx.array | None = None,
        cond_mask: mx.array | None = None,
        return_ith_layer: int | None = None,
        latent_cond: mx.array | None = None,
    ) -> dict[str, mx.array | None]:
        dtype = self.input_embed.proj[0].weight.dtype
        x = x.astype(dtype)
        text = text.astype(dtype)
        time = time.astype(dtype)
        batch = x.shape[0]
        if time.ndim == 0:
            time = mx.broadcast_to(time[None], (batch,))
        t = self.time_embed(time)
        if cond_mask is None:
            cond_mask = mx.arange(text.shape[1])[None, :] < text_len[:, None]
        text = self.text_embed(text, cond_mask)
        if self.text_conv:
            for layer in self.text_conv_layer:
                text = layer(text)
            text = mx.where(cond_mask[..., None], text, mx.zeros_like(text))
        x = self.input_embed(x, mask)
        if self.use_latent_condition and latent_cond is not None:
            latent_cond = self.latent_embed(latent_cond.astype(dtype), mask)
            x = self.latent_cond_embedder(
                mx.concatenate([x, latent_cond], axis=-1), mask
            )
        elif self.use_latent_condition:
            zeros = mx.zeros((batch, x.shape[1], self.config.dit_dim), dtype=dtype)
            x = self.latent_cond_embedder(mx.concatenate([x, zeros], axis=-1), mask)
        x_clone = x
        rope = self.rotary_embed(x.shape[1], x.dtype)
        cond_rope = self.rotary_embed(text.shape[1], text.dtype)
        if self.adaln_type == "global":
            if self.adaln_use_text_cond:
                text_mean = mx.sum(text, axis=1) / text_len.astype(dtype)[:, None]
                norm_cond = t + text_mean
            else:
                norm_cond = t
            adaln_mlp_out = self.adaln_global_mlp(norm_cond)
        else:
            norm_cond = None
            adaln_mlp_out = None
        hidden_state = None
        for index, block in enumerate(self.blocks, start=1):
            x = block(
                x=x,
                t=t,
                cond=text,
                mask=mask,
                cond_mask=cond_mask,
                rope=rope,
                cond_rope=cond_rope,
                adaln_global_out=adaln_mlp_out,
            )
            if return_ith_layer == index:
                hidden_state = x
                if self.long_skip:
                    x = x + x_clone
        if self.long_skip:
            x = x + x_clone
        x = self.norm_out(x, norm_cond if norm_cond is not None else t)
        return {"last_hidden_state": self.proj_out(x), "hidden_state": hidden_state}
