"""LTX feed-forward block — `GELU(approx="tanh")` + Linear.

Reference: `.references/DramaBox/ltx2/ltx_core/model/transformer/feed_forward.py`

Upstream layout:
    net = Sequential(
        GELUApprox(dim, dim * mult),   # net.0 — Linear named `.proj`
        Identity(),                    # net.1
        Linear(dim * mult, dim_out),   # net.2
    )

Saved key names:
    ff.net.0.proj.{weight, bias}   — input Linear inside GELUApprox
    ff.net.2.{weight, bias}        — output Linear

We mirror this exactly via a list-of-modules so MLX serializes children as
``net.0``, ``net.1``, ``net.2``.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class _GELUApprox(nn.Module):
    """Upstream `GELUApprox(dim_in, dim_out)`: Linear → tanh-approx GELU."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        return nn.gelu_approx(self.proj(x))


class LTXFeedForward(nn.Module):
    """Upstream `FeedForward(dim, dim_out, mult=4)`: GELU-approx → Linear."""

    def __init__(self, dim: int, dim_out: int, mult: int = 4):
        super().__init__()
        inner = int(dim * mult)
        # List-of-modules → serializes as net.0, net.1, net.2 matching upstream.
        self.net = [
            _GELUApprox(dim, inner),
            nn.Identity(),
            nn.Linear(inner, dim_out, bias=True),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.net[0](x)
        # net[1] is Identity — skip the call
        return self.net[2](x)


__all__ = ["LTXFeedForward"]
