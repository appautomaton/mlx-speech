from dataclasses import dataclass
from typing import Optional


@dataclass
class FishS2ProConfig:
    vocab_size: int = 4096
    num_codebooks: int = 10
    slow_ar_dim: int = 2048
    fast_ar_dim: int = 1024
    max_position_embeddings: int = 8192
    num_heads: int = 16
    num_layers: int = 30
    model_dir: str = "models/fish_s2_pro/mlx-int8"
    sample_rate: int = 22050

    @classmethod
    def from_huggingface(cls, repo_id: str):
        return cls(model_dir=repo_id)
