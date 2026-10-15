"""Fish Audio Drama 3 configuration (``model_type: fish_omni``).

One parser covers both the upstream release, whose routed experts are block FP8
(``quantization_config``), and the converted MLX checkpoint, whose routed experts
are affine-quantized (``quantization``). Every other weight is bfloat16 in both.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

MODEL_TYPE = "fish_omni"
TEXT_MODEL_TYPES = ("fish_omni_text", "qwen3_5_moe_text")
SPEECH_TEMPLATE_VERSION = "reference-bracket-v1"
FP8_QUANT_METHOD = "bsgemm-moe"
FP8_BLOCK_SIZE = 128

# Text flags fixed by the architecture: (key, default when absent, required value).
_FIXED_TEXT_FLAGS = (
    ("rotary_interleaved", False, False),
    ("use_qk_norm", True, True),
    ("attn_output_gate", True, True),
    ("layernorm_zero_centered_gamma", True, True),
    ("shared_expert_gate", True, True),
)
# attention_qk_norm is deliberately absent: the tower always applies q/k norm, but
# the upstream exporter never writes that field, so released configs say false.
_FIXED_AUDIO_DECODER_FLAGS = {
    "decoder_type": "codebook_mtp_shared",
    "rotary_interleaved": True,
    "attention_output_gate": True,
    "layernorm_zero_centered_gamma": True,
    "use_moe": False,
    "mtp_use_swa": False,
    "attention_qkv_bias": False,
    "attention_o_bias": False,
    "dropout": 0.0,
}


@dataclass(frozen=True, kw_only=True)
class FishDrama3TextConfig:
    """Qwen3.5-MoE hybrid backbone: gated DeltaNet and gated softmax attention."""

    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    partial_rotary_factor: float
    rms_norm_eps: float
    max_position_embeddings: int
    linear_conv_kernel_dim: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_num_key_heads: int
    linear_num_value_heads: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    norm_topk_prob: bool = True

    def __post_init__(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers or not set(
            self.layer_types
        ) <= {"linear_attention", "full_attention"}:
            raise ValueError(
                "layer_types must name linear_attention or full_attention for each "
                f"of the {self.num_hidden_layers} layers; got {self.layer_types!r}"
            )
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError(
                "num_attention_heads must be a multiple of num_key_value_heads"
            )
        if self.linear_num_value_heads % self.linear_num_key_heads:
            raise ValueError(
                "linear_num_value_heads must be a multiple of linear_num_key_heads"
            )
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok exceeds num_experts")
        if self.rotary_dim <= 0 or self.rotary_dim % 2:
            raise ValueError(
                f"rotary width must be positive and even; got {self.rotary_dim}"
            )

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FishDrama3TextConfig:
        if data.get("model_type") not in TEXT_MODEL_TYPES:
            raise ValueError(
                f"text_config.model_type={data.get('model_type')!r}; "
                f"expected one of {TEXT_MODEL_TYPES}"
            )
        if any(kind != "moe" for kind in data.get("mlp_layer_types") or ()):
            raise NotImplementedError("every text_config.mlp_layer_types must be moe")
        for key, default, required in _FIXED_TEXT_FLAGS:
            if data.get(key, default) is not required:
                raise NotImplementedError(
                    f"text_config.{key} must be {required}; got {data.get(key)!r}"
                )
        if data.get("tie_word_embeddings", False):
            raise NotImplementedError("the backbone keeps a separate lm_head")
        rope = data.get("rope_parameters") or data
        if rope.get("rope_type", rope.get("type", "default")) != "default":
            raise NotImplementedError("only the default rope_type is supported")

        num_layers = data["num_hidden_layers"]
        layer_types = data.get("layer_types") or [
            "full_attention"
            if (index + 1) % data["full_attention_interval"] == 0
            else "linear_attention"
            for index in range(num_layers)
        ]
        return cls(
            vocab_size=data["vocab_size"],
            hidden_size=data["hidden_size"],
            num_hidden_layers=num_layers,
            layer_types=tuple(layer_types),
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            rope_theta=float(rope["rope_theta"]),
            # Older configs keep the factor at the top level; 0.25 is the HF default.
            partial_rotary_factor=float(
                rope.get(
                    "partial_rotary_factor", data.get("partial_rotary_factor", 0.25)
                )
            ),
            rms_norm_eps=float(data["rms_norm_eps"]),
            max_position_embeddings=data["max_position_embeddings"],
            linear_conv_kernel_dim=data["linear_conv_kernel_dim"],
            linear_key_head_dim=data["linear_key_head_dim"],
            linear_value_head_dim=data["linear_value_head_dim"],
            linear_num_key_heads=data["linear_num_key_heads"],
            linear_num_value_heads=data["linear_num_value_heads"],
            num_experts=data["num_experts"],
            num_experts_per_tok=data["num_experts_per_tok"],
            moe_intermediate_size=data["moe_intermediate_size"],
            shared_expert_intermediate_size=data["shared_expert_intermediate_size"],
            norm_topk_prob=data.get("norm_topk_prob", True),
        )


@dataclass(frozen=True, kw_only=True)
class FishDrama3AudioDecoderConfig:
    """Codebook tower: predicts codes 1..N-1 of each frame from the backbone state."""

    dim: int
    intermediate_size: int
    n_head: int
    n_local_heads: int
    head_dim: int
    norm_eps: float
    rope_base: float
    rotary_percent: float
    vocab_size: int
    num_codebooks: int
    shared_enrichment_layers: int
    vq_backbone_codebooks: int

    def __post_init__(self) -> None:
        if (
            self.n_head % self.n_local_heads
            or not 0 < self.rotary_dim <= self.head_dim
            or self.rotary_dim % 2
        ):
            raise ValueError("invalid audio decoder attention geometry")
        if (
            self.num_codebooks < 2
            or not 0 <= self.vq_backbone_codebooks < self.num_codebooks
        ):
            raise ValueError("invalid codebook counts")
        if self.shared_enrichment_layers < 1:
            raise ValueError("shared_enrichment_layers must be >= 1")

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.rotary_percent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FishDrama3AudioDecoderConfig:
        for key, value in _FIXED_AUDIO_DECODER_FLAGS.items():
            if data.get(key, value) != value:
                raise NotImplementedError(
                    f"audio_decoder_config.{key}={data[key]!r} is not supported"
                )
        if data.get("text_dim", data["dim"]) != data["dim"]:
            raise ValueError("audio_decoder_config.text_dim must equal dim")
        if data.get("n_layer") != data["shared_enrichment_layers"]:
            raise ValueError("n_layer must equal shared_enrichment_layers")
        placement = data.get("vq_feature_placement")
        if placement not in ("backbone_partial", "mtp_shared_pre") or (
            placement == "mtp_shared_pre" and data["vq_backbone_codebooks"] != 0
        ):
            raise NotImplementedError(
                f"vq_feature_placement={placement!r} is not supported"
            )
        return cls(**{field.name: data[field.name] for field in fields(cls)})


@dataclass(frozen=True)
class ExpertQuantization:
    """MLX quantization of the routed experts in a converted checkpoint."""

    bits: int
    group_size: int
    mode: str = "affine"

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True, kw_only=True)
class FishDrama3Config:
    text: FishDrama3TextConfig
    audio_decoder: FishDrama3AudioDecoderConfig
    semantic_start_token_id: int
    semantic_end_token_id: int
    eos_token_id: int
    # Upstream release: routed experts are block FP8 with fp32 128x128 scales.
    fp8_experts: bool = False
    # Converted MLX checkpoint: routed experts are MLX-quantized.
    expert_quantization: ExpertQuantization | None = None

    def __post_init__(self) -> None:
        start, end = self.semantic_start_token_id, self.semantic_end_token_id
        if not 0 <= start <= end < self.text.vocab_size:
            raise ValueError("the semantic band must lie inside the text vocabulary")
        if start <= self.eos_token_id <= end:
            raise ValueError("eos_token_id lies inside the semantic band")
        if self.audio_decoder.dim != self.text.hidden_size:
            raise ValueError("audio decoder and backbone hidden sizes must match")
        if self.audio_decoder.vocab_size != end - start + 1:
            raise ValueError(
                "audio decoder vocabulary must match the semantic token band"
            )
        if self.fp8_experts and self.expert_quantization is not None:
            raise ValueError("experts cannot be both block FP8 and MLX-quantized")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FishDrama3Config:
        if data.get("model_type") != MODEL_TYPE:
            raise ValueError(
                f"model_type={data.get('model_type')!r}; expected {MODEL_TYPE!r}"
            )
        if data.get("audio_encoder_config") is not None:
            raise NotImplementedError("audio_encoder_config is not supported")
        if data.get("speech_template_version") != SPEECH_TEMPLATE_VERSION:
            raise NotImplementedError(
                f"speech_template_version={data.get('speech_template_version')!r}; "
                f"only {SPEECH_TEMPLATE_VERSION} is supported"
            )
        fp8 = data.get("quantization_config")
        if fp8 is not None and (
            fp8.get("quant_method") != FP8_QUANT_METHOD
            or fp8.get("weight_block_size") != [FP8_BLOCK_SIZE] * 2
        ):
            raise NotImplementedError(
                f"only {FP8_QUANT_METHOD} quantization with weight_block_size "
                f"[{FP8_BLOCK_SIZE}, {FP8_BLOCK_SIZE}] is supported"
            )
        mlx_quant = data.get("quantization")
        eos = data.get("eos_token_id", data["text_config"].get("eos_token_id"))
        if eos is None:
            raise ValueError("config has no eos_token_id")
        return cls(
            text=FishDrama3TextConfig.from_dict(data["text_config"]),
            audio_decoder=FishDrama3AudioDecoderConfig.from_dict(
                data["audio_decoder_config"]
            ),
            semantic_start_token_id=data["semantic_start_token_id"],
            semantic_end_token_id=data["semantic_end_token_id"],
            eos_token_id=eos,
            fp8_experts=fp8 is not None,
            expert_quantization=(
                None
                if mlx_quant is None
                else ExpertQuantization(
                    bits=int(mlx_quant["bits"]),
                    group_size=int(mlx_quant["group_size"]),
                    mode=str(mlx_quant.get("mode", "affine")),
                )
            ),
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> FishDrama3Config:
        with (Path(model_dir) / "config.json").open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
