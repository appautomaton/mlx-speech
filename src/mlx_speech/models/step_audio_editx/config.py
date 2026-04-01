"""Configuration helpers for Step-Audio-EditX Step1."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Step1Config:
    """MLX-facing config for the shipped Step1 checkpoint."""

    hidden_size: int = 3072
    intermediate_size: int = 8192
    num_attention_heads: int = 48
    num_attention_groups: int = 4
    num_hidden_layers: int = 32
    vocab_size: int = 74752
    rms_norm_eps: float = 1e-5
    bos_token_id: int = 1
    pad_token_id: int = 0
    eos_token_id: int = 3
    tie_word_embeddings: bool = False
    use_cache: bool = True
    max_seq_len: int = 32768
    model_type: str = "step1"
    architectures: tuple[str, ...] = ("Step1ForCausalLM",)
    torch_dtype: str = "bfloat16"
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_head_dim(self) -> int:
        return self.head_dim

    @property
    def kv_repeat(self) -> int:
        if self.num_attention_heads % self.num_attention_groups != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_attention_groups "
                f"({self.num_attention_heads} vs {self.num_attention_groups})."
            )
        return self.num_attention_heads // self.num_attention_groups

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Step1Config":
        raw = dict(payload)
        if "max_seq_len" not in raw and "max_position_embeddings" in raw:
            raw["max_seq_len"] = raw["max_position_embeddings"]

        if "architectures" in raw:
            arch = raw["architectures"]
            if isinstance(arch, str):
                raw["architectures"] = (arch,)
            elif isinstance(arch, list):
                raw["architectures"] = tuple(str(item) for item in arch)

        field_names = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        known = field_names - {"extra"}
        kwargs = {key: raw[key] for key in known if key in raw}
        extra = {key: value for key, value in raw.items() if key not in known}
        return cls(**kwargs, extra=extra)

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "Step1Config":
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with config_path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_attention_groups": self.num_attention_groups,
            "num_hidden_layers": self.num_hidden_layers,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "tie_word_embeddings": self.tie_word_embeddings,
            "use_cache": self.use_cache,
            "max_seq_len": self.max_seq_len,
            "model_type": self.model_type,
            "architectures": list(self.architectures),
            "torch_dtype": self.torch_dtype,
        }
        payload.update(self.extra)
        return payload
