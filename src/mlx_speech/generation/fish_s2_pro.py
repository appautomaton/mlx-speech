from dataclasses import dataclass
from typing import Optional

import mlx.core as mx


@dataclass
class FishS2ProOutput:
    """Output from Fish S2 Pro generation."""

    waveform: Optional[mx.array]
    sample_rate: int = 22050
    generated_tokens: int = 0


def generate_fish_s2_pro(
    text: str,
    *,
    model_dir: str = "models/fish_s2_pro/mlx-int8",
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.8,
    top_k: int = 50,
) -> FishS2ProOutput:
    """Generate speech from text using Fish S2 Pro.

    Args:
        text: Input text (can include inline tags like [whisper], [excited])
        model_dir: Path to model checkpoint
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling

    Returns:
        FishS2ProOutput with waveform
    """
    return FishS2ProOutput(
        waveform=None,
        sample_rate=22050,
        generated_tokens=0,
    )
