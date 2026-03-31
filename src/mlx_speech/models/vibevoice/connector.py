"""Speech connectors for VibeVoice Large."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class SpeechConnector(nn.Module):
    """Projects speech latents into the LM hidden space.

    Architecture: Linear → RMSNorm → Linear (matches upstream SpeechConnector).
    """

    def __init__(self, input_dim: int, output_dim: int, eps: float = 1e-6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim, bias=True)
        self.norm = nn.RMSNorm(output_dim, eps=eps)
        self.fc2 = nn.Linear(output_dim, output_dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        """Project features to LM hidden size.

        Args:
            x: (..., input_dim)

        Returns:
            (..., output_dim)
        """
        return self.fc2(self.norm(self.fc1(x)))
