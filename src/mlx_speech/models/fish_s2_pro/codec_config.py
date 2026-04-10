from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FishCodecConfig:
    sample_rate: int = 44100
    latent_dim: int = 1024
    semantic_codebook_size: int = 4096
    n_codebooks: int = 9
    decoder_dim: int = 1536
    encoder_dim: int = 64

    def to_dict(self) -> dict[str, int]:
        return asdict(self)
