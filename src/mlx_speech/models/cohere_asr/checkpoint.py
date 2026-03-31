"""Checkpoint loading and quantization for CohereAsr."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints.sharded import load_state_dict
from .config import CohereAsrConfig


@dataclass(frozen=True)
class QuantizationConfig:
    bits: int = 8
    group_size: int = 64
    mode: str = "affine"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuantizationConfig":
        return cls(bits=int(d["bits"]), group_size=int(d["group_size"]), mode=str(d.get("mode", "affine")))

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True)
class AlignmentReport:
    missing_in_model: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple, tuple], ...]

    @property
    def is_exact_match(self) -> bool:
        return not self.missing_in_model and not self.missing_in_checkpoint and not self.shape_mismatches


@dataclass(frozen=True)
class CohereAsrCheckpoint:
    model_dir: Path
    config: CohereAsrConfig
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    skipped_keys: tuple[str, ...]
    renamed_keys: tuple[tuple[str, str], ...]


# ---------------------------------------------------------------------------
# Key sanitization — maps NeMo checkpoint keys → MLX module-tree paths
# ---------------------------------------------------------------------------

# Skip these keys entirely (never loaded into the MLX model)
_SKIP_RE = re.compile(
    r"^preprocessor\."            # feature-extraction buffers — we compute our own
    r"|\.num_batches_tracked$"     # BatchNorm bookkeeping scalar, not a parameter
)

# Pattern-based renames: (compiled_regex, replacement_template)
# Applied in order; first match wins.
_RENAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ------------------------------------------------------------------
    # Encoder subsampling (pre_encode)
    # ------------------------------------------------------------------
    # First conv: standard Conv2d(1 → ch)
    (re.compile(r"^encoder\.pre_encode\.conv\.0\.(weight|bias)$"),
     r"encoder.subsampling.conv0.\1"),
    # Depthwise Conv2d layers (groups=ch), stored as plain arrays
    (re.compile(r"^encoder\.pre_encode\.conv\.2\.weight$"),
     r"encoder.subsampling.dw_weight_0"),
    (re.compile(r"^encoder\.pre_encode\.conv\.2\.bias$"),
     r"encoder.subsampling.dw_bias_0"),
    (re.compile(r"^encoder\.pre_encode\.conv\.5\.weight$"),
     r"encoder.subsampling.dw_weight_1"),
    (re.compile(r"^encoder\.pre_encode\.conv\.5\.bias$"),
     r"encoder.subsampling.dw_bias_1"),
    # Pointwise 1×1 Conv2d
    (re.compile(r"^encoder\.pre_encode\.conv\.3\.(weight|bias)$"),
     r"encoder.subsampling.pw_convs.0.\1"),
    (re.compile(r"^encoder\.pre_encode\.conv\.6\.(weight|bias)$"),
     r"encoder.subsampling.pw_convs.1.\1"),
    # Output linear
    (re.compile(r"^encoder\.pre_encode\.out\.(weight|bias)$"),
     r"encoder.subsampling.linear.\1"),

    # ------------------------------------------------------------------
    # Encoder attention (per layer)
    # ------------------------------------------------------------------
    (re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.linear_q\.(weight|bias)$"),
     r"\1.self_attn.q_proj.\2"),
    (re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.linear_k\.(weight|bias)$"),
     r"\1.self_attn.k_proj.\2"),
    (re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.linear_v\.(weight|bias)$"),
     r"\1.self_attn.v_proj.\2"),
    (re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.linear_out\.(weight|bias)$"),
     r"\1.self_attn.o_proj.\2"),
    (re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.linear_pos\.weight$"),
     r"\1.self_attn.relative_k_proj.weight"),
    (re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.pos_bias_u$"),
     r"\1.self_attn.bias_u"),
    (re.compile(r"^(encoder\.layers\.\d+)\.self_attn\.pos_bias_v$"),
     r"\1.self_attn.bias_v"),

    # ------------------------------------------------------------------
    # Encoder conv module (per layer)
    # ------------------------------------------------------------------
    # Depthwise Conv1d — stored as plain arrays (no leading underscore for MLX param tree)
    (re.compile(r"^(encoder\.layers\.\d+)\.conv\.depthwise_conv\.weight$"),
     r"\1.conv.dw_weight"),
    (re.compile(r"^(encoder\.layers\.\d+)\.conv\.depthwise_conv\.bias$"),
     r"\1.conv.dw_bias"),
    # BatchNorm → MLX nn.BatchNorm (norm)
    (re.compile(r"^(encoder\.layers\.\d+)\.conv\.batch_norm\.(weight|bias|running_mean|running_var)$"),
     r"\1.conv.norm.\2"),

    # ------------------------------------------------------------------
    # Top-level encoder→decoder projection
    # ------------------------------------------------------------------
    (re.compile(r"^encoder_decoder_proj\.(weight|bias)$"),
     r"decoder.proj.\1"),

    # ------------------------------------------------------------------
    # Decoder self-attention (first_sub_layer)
    # ------------------------------------------------------------------
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.first_sub_layer\.query_net\.(weight|bias)$"),
     r"decoder.layers.\1.self_attn.q_proj.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.first_sub_layer\.key_net\.(weight|bias)$"),
     r"decoder.layers.\1.self_attn.k_proj.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.first_sub_layer\.value_net\.(weight|bias)$"),
     r"decoder.layers.\1.self_attn.v_proj.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.first_sub_layer\.out_projection\.(weight|bias)$"),
     r"decoder.layers.\1.self_attn.o_proj.\2"),

    # ------------------------------------------------------------------
    # Decoder cross-attention (second_sub_layer)
    # ------------------------------------------------------------------
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.second_sub_layer\.query_net\.(weight|bias)$"),
     r"decoder.layers.\1.encoder_attn.q_proj.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.second_sub_layer\.key_net\.(weight|bias)$"),
     r"decoder.layers.\1.encoder_attn.k_proj.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.second_sub_layer\.value_net\.(weight|bias)$"),
     r"decoder.layers.\1.encoder_attn.v_proj.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.second_sub_layer\.out_projection\.(weight|bias)$"),
     r"decoder.layers.\1.encoder_attn.o_proj.\2"),

    # ------------------------------------------------------------------
    # Decoder MLP (third_sub_layer)
    # ------------------------------------------------------------------
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.third_sub_layer\.dense_in\.(weight|bias)$"),
     r"decoder.layers.\1.mlp.fc1.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.third_sub_layer\.dense_out\.(weight|bias)$"),
     r"decoder.layers.\1.mlp.fc2.\2"),

    # ------------------------------------------------------------------
    # Decoder layer norms (per layer)
    # ------------------------------------------------------------------
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_1\.(weight|bias)$"),
     r"decoder.layers.\1.input_layernorm.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_2\.(weight|bias)$"),
     r"decoder.layers.\1.post_attention_layernorm.\2"),
    (re.compile(r"^transf_decoder\._decoder\.layers\.(\d+)\.layer_norm_3\.(weight|bias)$"),
     r"decoder.layers.\1.final_layernorm.\2"),

    # ------------------------------------------------------------------
    # Decoder top-level
    # ------------------------------------------------------------------
    (re.compile(r"^transf_decoder\._decoder\.final_layer_norm\.(weight|bias)$"),
     r"decoder.norm.\1"),
    (re.compile(r"^transf_decoder\._embedding\.token_embedding\.weight$"),
     r"decoder.embed_tokens.weight"),
    # pos_enc is a precomputed table — loaded into nn.Embedding.weight
    (re.compile(r"^transf_decoder\._embedding\.position_embedding\.pos_enc$"),
     r"decoder.pos_emb.weight"),
    (re.compile(r"^transf_decoder\._embedding\.layer_norm\.(weight|bias)$"),
     r"decoder.embedding_layernorm.\1"),

    # ------------------------------------------------------------------
    # LM head (weight is tied to embed_tokens; we load both separately)
    # ------------------------------------------------------------------
    (re.compile(r"^log_softmax\.mlp\.layer0\.(weight|bias)$"),
     r"proj_out.\1"),
]


def _apply_rename(key: str) -> str:
    """Return the renamed key, or the original key if no pattern matches."""
    for pattern, replacement in _RENAME_PATTERNS:
        if pattern.match(key):
            return pattern.sub(replacement, key)
    return key


# Matches any key whose final component is "weight" (possibly with a numeric suffix).
# Covers: `.weight`, `_weight`, `_weight_0`, `_dw_weight`, `_dw_weight_0`, etc.
_WEIGHT_KEY_RE = re.compile(r"(?:[._])weight(?:_\d+)?$")


def _maybe_transpose(key: str, value: mx.array) -> mx.array:
    """Transpose convolution weights from PyTorch layout to MLX layout.

    PyTorch Conv2d: (C_out, C_in, H, W) → MLX: (C_out, H, W, C_in)
    PyTorch Conv1d: (C_out, C_in, L)    → MLX: (C_out, L, C_in)

    Applies to any key whose suffix looks like a weight (see _WEIGHT_KEY_RE).
    All ndim≥3 tensors in this model are convolution weights; Linear is 2D.
    """
    if not _WEIGHT_KEY_RE.search(key):
        return value
    if value.ndim == 4:
        return value.transpose(0, 2, 3, 1)
    if value.ndim == 3:
        return value.transpose(0, 2, 1)
    return value


def sanitize_state_dict(
    weights: dict[str, mx.array],
) -> tuple[dict[str, mx.array], tuple[str, ...], tuple[tuple[str, str], ...]]:
    """Remap NeMo checkpoint keys to the MLX module tree.

    Returns:
        sanitized:  dict ready for model.load_weights()
        skipped:    keys that were dropped
        renamed:    (original_key, new_key) for every key that changed
    """
    sanitized: dict[str, mx.array] = {}
    skipped: list[str] = []
    renamed: list[tuple[str, str]] = []

    for key, value in weights.items():
        if _SKIP_RE.search(key):
            skipped.append(key)
            continue

        new_key = _apply_rename(key)
        value = _maybe_transpose(new_key, value)

        if new_key in sanitized:
            raise ValueError(f"Duplicate key after sanitization: {new_key!r}")
        sanitized[new_key] = value
        if new_key != key:
            renamed.append((key, new_key))

    return sanitized, tuple(skipped), tuple(renamed)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _is_mlx_format(files: tuple[Path, ...]) -> bool:
    """Return True if any safetensors file has metadata format=='mlx'."""
    try:
        from safetensors import safe_open
        for f in files:
            with safe_open(str(f), framework="numpy") as h:
                meta = h.metadata() or {}
                if meta.get("format") == "mlx":
                    return True
    except Exception:
        pass
    return False


def load_cohere_asr_checkpoint(model_dir: str | Path) -> CohereAsrCheckpoint:
    model_dir = Path(model_dir)
    config = CohereAsrConfig.from_path(model_dir)
    loaded = load_state_dict(model_dir)

    if _is_mlx_format(loaded.files):
        # Already-converted MLX checkpoint: weights are in MLX layout, no sanitize needed.
        return CohereAsrCheckpoint(
            model_dir=model_dir,
            config=config,
            state_dict=loaded.weights,
            source_files=loaded.files,
            skipped_keys=(),
            renamed_keys=(),
        )

    state_dict, skipped, renamed = sanitize_state_dict(loaded.weights)
    return CohereAsrCheckpoint(
        model_dir=model_dir,
        config=config,
        state_dict=state_dict,
        source_files=loaded.files,
        skipped_keys=skipped,
        renamed_keys=renamed,
    )


def validate_checkpoint_against_model(
    model: nn.Module,
    checkpoint: CohereAsrCheckpoint,
) -> AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    mismatches = []
    for key in sorted(model_keys & checkpoint_keys):
        ms = tuple(int(d) for d in model_params[key].shape)
        cs = tuple(int(d) for d in checkpoint.state_dict[key].shape)
        if ms != cs:
            mismatches.append((key, ms, cs))

    return AlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(mismatches),
    )


def load_checkpoint_into_model(
    model: nn.Module,
    checkpoint: CohereAsrCheckpoint,
    *,
    strict: bool = True,
) -> AlignmentReport:
    report = validate_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            f"Checkpoint alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only, "
            f"{len(report.missing_in_checkpoint)} model-only, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return report


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_cohere_asr_model(
    model: nn.Module,
    quantization: QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> nn.Module:
    quantized_keys = set(state_dict) if state_dict is not None else None

    def should_quantize(path: str, module: Any) -> bool:
        if not (hasattr(module, "weight") and hasattr(module, "to_quantized")):
            return False
        if module.weight.shape[-1] % quantization.group_size != 0:
            return False
        if quantized_keys is None:
            return True
        return f"{path}.scales" in quantized_keys

    nn.quantize(
        model,
        group_size=quantization.group_size,
        bits=quantization.bits,
        mode=quantization.mode,
        class_predicate=should_quantize,
    )
    return model


def save_cohere_asr_model(
    model: nn.Module,
    model_dir: str | Path,
    *,
    config: CohereAsrConfig,
    quantization: QuantizationConfig | None = None,
) -> Path:
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights = tree_flatten(model.parameters(), destination={})
    mx.eval(list(weights.values()))
    mx.save_safetensors(
        str(output_dir / "model.safetensors"),
        weights,
        metadata={"format": "mlx"},
    )

    payload = config.to_dict()
    if quantization is not None:
        payload["quantization"] = quantization.to_dict()
    with (output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    return output_dir


def get_quantization_config(config: CohereAsrConfig) -> QuantizationConfig | None:
    q = config.extra.get("quantization")
    if q is None:
        return None
    return QuantizationConfig.from_dict(q)
