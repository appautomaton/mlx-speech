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


# What dense_int8 packs: the BF16 projections every frame reads in full. In the
# backbone, attention, Gated DeltaNet and shared-expert projections; in the audio
# decoder, every tower layer. The routers, the DeltaNet decay gates (in_proj_a,
# in_proj_b), embeddings, lm_head and the codebook heads stay BF16.
DENSE_INT8_GROUP_SIZE = 64
_BACKBONE_DENSE = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".linear_attn.in_proj_qkv",
    ".linear_attn.in_proj_z",
    ".linear_attn.out_proj",
    ".mlp.shared_expert.gate_proj",
    ".mlp.shared_expert.up_proj",
    ".mlp.shared_expert.down_proj",
)
_TOWER_DENSE = (
    ".eh_proj",
    ".transformer_layer.attention.wqkv",
    ".transformer_layer.attention.wo",
    ".transformer_layer.feed_forward.w1",
    ".transformer_layer.feed_forward.w2",
    ".transformer_layer.feed_forward.w3",
)


def is_dense_int8_target(path: str, module: nn.Module) -> bool:
    """Whether dense_int8 packs this module."""
    if type(module) is not nn.Linear:
        return False
    if path.startswith("backbone.layers."):
        return path.endswith(_BACKBONE_DENSE)
    if path.startswith(("audio_decoder.shared_layers.", "audio_decoder.residual_layers.")):
        return path.endswith(_TOWER_DENSE)
    return False


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


def load_fish_drama_3_model(
    model_dir: str | Path, *, materialize: bool = True, dense_int8: bool = False
):
    """Load every weight strictly; materialize reads them all into memory now
    rather than on first use. dense_int8 packs the dense BF16 projections as
    affine int8 (groups of 64) once loaded: the checkpoint is unchanged."""
    config = FishDrama3Config.from_path(model_dir)
    model = FishDrama3Model(config)
    model.load_weights(list(load_state_dict(model_dir).weights.items()), strict=True)
    if dense_int8:
        nn.quantize(
            model,
            group_size=DENSE_INT8_GROUP_SIZE,
            bits=8,
            mode="affine",
            class_predicate=is_dense_int8_target,
        )
    if materialize:
        mx.eval(model.parameters())
    return model
