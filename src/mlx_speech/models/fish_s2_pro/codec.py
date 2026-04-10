from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx


REQUIRED_CODEC_ASSETS = ("config.json", "model.safetensors")


class MissingCodecAssetError(FileNotFoundError):
    pass


def validate_codec_assets(codec_dir: str | Path) -> Path:
    resolved = Path(codec_dir)
    missing_assets = [
        asset_name
        for asset_name in REQUIRED_CODEC_ASSETS
        if not (resolved / asset_name).is_file()
    ]
    if missing_assets:
        missing_list = ", ".join(missing_assets)
        raise MissingCodecAssetError(
            f"Fish S2 codec assets not found at {resolved}. Missing required files: {missing_list}. Provide a local MLX Fish codec directory."
        )
    return resolved


@dataclass
class FishS2Codec:
    model: object
    sample_rate: int = 44100

    @classmethod
    def from_dir(cls, codec_dir: str | Path) -> "FishS2Codec":
        resolved = validate_codec_assets(codec_dir)
        instance = cls(model=None)
        instance.model = instance._load_model(resolved)
        instance.sample_rate = getattr(
            instance.model, "sample_rate", instance.sample_rate
        )
        return instance

    def _load_model(self, codec_dir: Path):
        from .codec_model import FishCodecModel

        return FishCodecModel.from_dir(codec_dir)

    def decode(self, codes: mx.array) -> mx.array:
        return self.model.decode(codes)
