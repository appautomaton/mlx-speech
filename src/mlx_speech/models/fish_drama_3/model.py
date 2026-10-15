"""Drama 3 model: the text backbone and the audio decoder, loaded from a converted
MLX checkpoint (scripts/convert/fish_drama_3.py)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from ...checkpoints.sharded import load_state_dict
from .audio_decoder import FishDrama3AudioDecoder
from .backbone import FishDrama3Backbone
from .config import FishDrama3Config


class FishDrama3Model(nn.Module):
    def __init__(self, config: FishDrama3Config):
        super().__init__()
        if config.expert_quantization is None:
            raise ValueError(
                "Drama 3 runs from a converted MLX checkpoint; convert the upstream "
                "release with scripts/convert/fish_drama_3.py"
            )
        self.config = config
        self.backbone = FishDrama3Backbone(config.text, config.expert_quantization)
        self.audio_decoder = FishDrama3AudioDecoder(config)


def load_fish_drama_3_model(model_dir: str | Path, *, materialize: bool = True):
    """Load every weight strictly; materialize reads them all into memory now
    rather than on first use."""
    config = FishDrama3Config.from_path(model_dir)
    model = FishDrama3Model(config)
    model.load_weights(list(load_state_dict(model_dir).weights.items()), strict=True)
    if materialize:
        mx.eval(model.parameters())
    return model
