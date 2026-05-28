"""End-of-clip silence-prior fix.

For target latents with `T > 513`, the upstream code rewrites the values at
latent frame indices 512 and 513 by linearly interpolating between frames
511 and 514. This trims off a high-frequency artifact that the DiT's RoPE
produces at very long sequences.

Reference: `.references/DramaBox/src/inference_server.py:411-429`
"""

from __future__ import annotations

import mlx.core as mx


def silence_prior_fix(latent_4d: mx.array) -> mx.array:
    """Linearly interpolate frames 512 / 513 from frames 511 and 514.

    Args:
        latent_4d: ``[B, C, T, F]`` un-patchified latent.

    Returns:
        Same shape; only frames 512 / 513 modified (if `T > 513`); otherwise
        the input is returned unchanged.
    """
    if latent_4d.shape[2] <= 513:
        return latent_4d

    # Linear interp at index 512 (1/3 between 511 and 514) and 513 (2/3)
    a = latent_4d[:, :, 511, :]
    b = latent_4d[:, :, 514, :]
    out = latent_4d
    # Build with concatenate to avoid in-place ops
    pre = out[:, :, :512, :]
    f512 = (a * (2 / 3) + b * (1 / 3))[:, :, None, :]
    f513 = (a * (1 / 3) + b * (2 / 3))[:, :, None, :]
    post = out[:, :, 514:, :]
    return mx.concatenate([pre, f512, f513, post], axis=2)


__all__ = ["silence_prior_fix"]
