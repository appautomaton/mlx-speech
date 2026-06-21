"""Bidirectional Mamba blocks for the RE-USE / SEMamba port (pure MLX).

Mirrors `.references/RE-USE/models/mamba_block2_SEMamba.py` and the single-
direction `mamba_ssm.Mamba` non-fast-path forward in
`.references/mamba_ssm/mamba_simple.py:161-205`.

Module/param names match the torch checkpoint keys so weight remapping in
`loader.py` stays mechanical:

    <dir>.in_proj.weight  <dir>.conv1d.{weight,bias}  <dir>.x_proj.weight
    <dir>.dt_proj.{weight,bias}  <dir>.A_log  <dir>.D  <dir>.out_proj.weight
    output_proj.{weight,bias}  norm.{weight,bias}

where <dir> is ``forward_blocks`` or ``backward_blocks``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .scan import selective_scan


class MambaSSM(nn.Module):
    """One direction of a Mamba SSM (the upstream ``mamba_ssm.Mamba`` module).

    Forward only. The bidirectional block obtains the backward direction by
    flipping the whole input in time (the depthwise causal conv included) and
    flipping the output back, exactly as the reference does. This module never
    reverses internally, so its conv1d runs in the same time order as its scan.
    Input/output are ``[B, L, d_model]``.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = (d_model + 15) // 16  # ceil(d_model / 16); "auto"

        # x -> (x_in, z), each d_inner. No bias (upstream bias=False).
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # Depthwise causal conv over d_inner channels, kernel d_conv.
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        # x -> (dt, B, C). No bias.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        # dt_rank -> d_inner. Bias is the scan's delta_bias.
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.A_log = mx.zeros((self.d_inner, d_state))
        self.D = mx.ones((self.d_inner,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, L, d_model]
        batch, length, _ = x.shape
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_in, z = mx.split(xz, [self.d_inner], axis=-1)  # each [B, L, d_inner]

        # Depthwise causal conv on x_in. MLX Conv1d takes [B, L, C]; padding
        # d_conv-1 makes it left+right, so trim the last d_conv-1 to stay causal.
        x_conv = self.conv1d(x_in)[:, :length, :]  # [B, L, d_inner]
        x_conv = nn.silu(x_conv)

        # x_proj -> (dt, B, C).
        x_dbl = self.x_proj(x_conv)  # [B, L, dt_rank + 2*d_state]
        dt, B_var, C_var = mx.split(
            x_dbl, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )
        # dt_proj.weight @ dt (no bias here; bias goes to the scan as delta_bias).
        delta = dt @ self.dt_proj.weight.T  # [B, L, d_inner]

        A = -mx.exp(self.A_log.astype(mx.float32))  # [d_inner, d_state]

        # The scan wants [B, d_inner, L] for u/delta/z and [B, d_state, L] for B/C.
        u = mx.transpose(x_conv, (0, 2, 1))
        delta = mx.transpose(delta, (0, 2, 1))
        z_t = mx.transpose(z, (0, 2, 1))
        B_t = mx.transpose(B_var, (0, 2, 1))
        C_t = mx.transpose(C_var, (0, 2, 1))

        y = selective_scan(
            u,
            delta,
            A,
            B_t,
            C_t,
            D=self.D,
            z=z_t,
            delta_bias=self.dt_proj.bias,
            delta_softplus=True,
        )  # [B, d_inner, L]

        y = mx.transpose(y, (0, 2, 1))  # [B, L, d_inner]
        return self.out_proj(y)  # [B, L, d_model]


class MambaBlock(nn.Module):
    """Bidirectional Mamba block (combine rule from the slice-1 notes).

        out_fw = forward_blocks(x) + x
        out_bw = flip(backward_blocks(flip(x)) + flip(x))   (whole module on flipped x)
        out    = output_proj(cat([out_fw, out_bw], -1))
        return LayerNorm(out)

    The backward branch flips x in time, runs the *entire* backward module
    (including its depthwise causal conv) on the reversed sequence, then flips
    back, exactly matching `mamba_block2_SEMamba.py:MambaBlock.forward`.
    Input/output ``[B, L, d_model]``.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.forward_blocks = MambaSSM(d_model, d_state, d_conv, expand)
        self.backward_blocks = MambaSSM(d_model, d_state, d_conv, expand)
        self.output_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def _flip_time(x: mx.array) -> mx.array:
        # Reverse the L axis (axis 1). MLX 0.31 has no mx.flip and mishandles
        # elementwise ops on reversed-stride views, so materialize contiguous.
        return mx.contiguous(x[:, ::-1, :])

    def __call__(self, x: mx.array) -> mx.array:
        out_fw = self.forward_blocks(x) + x
        x_rev = self._flip_time(x)
        out_bw = self._flip_time(self.backward_blocks(x_rev) + x_rev)
        out = mx.concatenate([out_fw, out_bw], axis=-1)
        out = self.output_proj(out)
        return self.norm(out)


class TFMambaBlock(nn.Module):
    """Time-frequency Mamba block.

    Mirrors `.references/RE-USE/models/mamba_block2_SEMamba.py:TFMambaBlock`.
    Input/output ``[B, C, T, F]``: time_mamba over ``(b*f, t, c)`` then
    freq_mamba over ``(b*t, f, c)``, each with a residual.
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.time_mamba = MambaBlock(d_model, d_state, d_conv, expand)
        self.freq_mamba = MambaBlock(d_model, d_state, d_conv, expand)

    def __call__(self, x: mx.array) -> mx.array:
        b, c, t, f = x.shape
        # (b, c, t, f) -> (b, f, t, c) -> (b*f, t, c)
        xt = mx.transpose(x, (0, 3, 2, 1)).reshape(b * f, t, c)
        xt = self.time_mamba(xt) + xt
        # (b*f, t, c) -> (b, f, t, c) -> (b, t, f, c) -> (b*t, f, c)
        xf = mx.transpose(xt.reshape(b, f, t, c), (0, 2, 1, 3)).reshape(b * t, f, c)
        xf = self.freq_mamba(xf) + xf
        # (b*t, f, c) -> (b, t, f, c) -> (b, c, t, f)
        out = mx.transpose(xf.reshape(b, t, f, c), (0, 3, 1, 2))
        return out


__all__ = ["MambaSSM", "MambaBlock", "TFMambaBlock"]
