"""DiT config — values are fixed for the DramaBox dev checkpoint."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiTConfig:
    """Static DiT config for DramaBox.

    All values were resolved from `dramabox-dit-v1.safetensors`'s
    `__metadata__.config` block.
    """

    # Audio-only path
    audio_in_channels: int = 128
    audio_out_channels: int = 128
    audio_num_attention_heads: int = 32
    audio_attention_head_dim: int = 64
    audio_cross_attention_dim: int = 2048
    audio_positional_embedding_max_pos: tuple[int, ...] = (20,)
    num_layers: int = 48
    norm_eps: float = 1e-6

    # AdaLN
    cross_attention_adaln: bool = True
    apply_gated_attention: bool = True

    # RoPE
    rope_type: str = "split"
    positional_embedding_theta: float = 10000.0

    # Timestep encoding
    timestep_scale_multiplier: int = 1000

    # Sequence/grid options
    use_middle_indices_grid: bool = True

    @property
    def audio_inner_dim(self) -> int:
        return self.audio_num_attention_heads * self.audio_attention_head_dim


__all__ = ["DiTConfig"]
