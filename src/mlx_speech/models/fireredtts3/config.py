"""Configuration contract for flat FireRedTTS3 Base MLX artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODEL_TYPE = "fireredtts3_base"
FORMAT_VERSION = 1
ARTIFACT_FILES = {
    "core": "core.safetensors",
    "redae": "redae.safetensors",
    "speaker": "speaker.safetensors",
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class FireRedTTS3Config:
    """Validated root configuration for a FireRedTTS3 Base artifact."""

    payload: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FireRedTTS3Config":
        config = cls(dict(payload))
        config.validate()
        return config

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "FireRedTTS3Config":
        path = Path(model_dir) / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"FireRedTTS3 config not found: {path}")
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("FireRedTTS3 config.json must contain an object")
        return cls.from_dict(payload)

    @property
    def model_type(self) -> str:
        return str(self.payload["model_type"])

    @property
    def sample_rate(self) -> int:
        return int(self.payload["redae"]["audio_sample_rate"])

    @property
    def files(self) -> dict[str, str]:
        return dict(self.payload["files"])

    @property
    def core(self) -> dict[str, Any]:
        return dict(self.payload["core"])

    @property
    def redae(self) -> dict[str, Any]:
        return dict(self.payload["redae"])

    @property
    def qwen(self) -> dict[str, Any]:
        return dict(self.payload["qwen"])

    @property
    def speaker(self) -> dict[str, Any]:
        return dict(self.payload["speaker"])

    def validate(self) -> None:
        if self.payload.get("model_type") != MODEL_TYPE:
            raise ValueError(
                f"expected model_type {MODEL_TYPE!r}, got "
                f"{self.payload.get('model_type')!r}"
            )
        if self.payload.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"unsupported FireRedTTS3 format_version: "
                f"{self.payload.get('format_version')!r}"
            )
        if self.payload.get("precision") != "bfloat16":
            raise ValueError("FireRedTTS3 Base artifact precision must be bfloat16")
        files = self.payload.get("files")
        if files != ARTIFACT_FILES:
            raise ValueError(
                f"FireRedTTS3 component files must be {ARTIFACT_FILES!r}, got {files!r}"
            )
        tokenizer_files = tuple(self.payload.get("tokenizer_files", ()))
        if tokenizer_files != TOKENIZER_FILES:
            raise ValueError(
                "FireRedTTS3 tokenizer_files must name the three flat root files"
            )
        required_sections = ("core", "redae", "qwen", "speaker", "sources")
        missing = [name for name in required_sections if not self.payload.get(name)]
        if missing:
            raise ValueError(f"FireRedTTS3 config missing sections: {missing}")
        if int(self.payload["redae"].get("audio_sample_rate", 0)) != 24_000:
            raise ValueError("FireRedTTS3 RedAE sample rate must be 24000")
        if int(self.payload["redae"].get("bottleneck_dim", 0)) != 64:
            raise ValueError("FireRedTTS3 RedAE bottleneck_dim must be 64")
        if int(self.payload["core"].get("patch_size", 0)) != 4:
            raise ValueError("FireRedTTS3 Base patch_size must be 4")
        if self.payload["core"].get("dtype") != "bfloat16":
            raise ValueError("FireRedTTS3 Base core dtype must be bfloat16")
        if self.payload["redae"].get("dtype") != "bfloat16":
            raise ValueError("FireRedTTS3 RedAE dtype must be bfloat16")
        if int(self.payload["speaker"].get("embedding_size", 0)) != 512:
            raise ValueError("FireRedTTS3 speaker embedding_size must be 512")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)


__all__ = [
    "ARTIFACT_FILES",
    "FORMAT_VERSION",
    "MODEL_TYPE",
    "TOKENIZER_FILES",
    "FireRedTTS3Config",
]
