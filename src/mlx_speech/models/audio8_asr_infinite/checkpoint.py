"""Strict BF16 checkpoint conversion for Audio8-ASR-Infinite."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from ...checkpoints.sharded import load_state_dict
from .config import Audio8Config


CONV1D_WEIGHT_KEYS = frozenset(
    {"audio_tower.embedder.conv1.weight", "audio_tower.embedder.conv2.weight"}
)
TIED_LM_HEAD_KEY = "language_model.lm_head.weight"
GENERATED_MODEL_ONLY_KEYS = frozenset(
    {"time_embedding.inv_freq", "audio_tower.rotary_emb.inv_freq"}
)
SUPPORTING_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "generation_config.json",
)


@dataclass(frozen=True)
class Audio8QuantizationConfig:
    bits: int = 8
    group_size: int = 64
    mode: str = "affine"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Audio8QuantizationConfig":
        return cls(
            bits=int(payload["bits"]),
            group_size=int(payload["group_size"]),
            mode=str(payload.get("mode", "affine")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"bits": self.bits, "group_size": self.group_size, "mode": self.mode}


@dataclass(frozen=True)
class Audio8Checkpoint:
    model_dir: Path
    config: Audio8Config
    state_dict: dict[str, mx.array]
    source_files: tuple[Path, ...]
    transposed_keys: tuple[str, ...]


@dataclass(frozen=True)
class Audio8ConversionReport:
    output_file: Path
    tensor_count: int
    transposed_keys: tuple[str, ...]
    copied_files: tuple[Path, ...]


@dataclass(frozen=True)
class Audio8AlignmentReport:
    checkpoint_only: tuple[str, ...]
    model_only: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]


def expected_audio8_tensors(config: Audio8Config) -> dict[str, tuple[int, ...]]:
    audio = config.audio_config
    text = config.text_config
    expected = {
        "audio_tower.embedder.conv1.weight": (audio.hidden_size, 3, audio.num_mel_bins),
        "audio_tower.embedder.conv1.bias": (audio.hidden_size,),
        "audio_tower.embedder.conv2.weight": (audio.hidden_size, 3, audio.hidden_size),
        "audio_tower.embedder.conv2.bias": (audio.hidden_size,),
        "audio_tower.norm.weight": (audio.hidden_size,),
        "multi_modal_projector.linear_1.weight": (text.hidden_size, config.projection_size),
        "multi_modal_projector.linear_2.weight": (text.hidden_size, text.hidden_size),
        "language_model.model.embed_tokens.weight": (text.vocab_size, text.hidden_size),
        "language_model.model.norm.weight": (text.hidden_size,),
    }
    if config.use_frame_len_embedding:
        expected["frame_len_embedding.weight"] = (
            len(config.supported_frame_lens), text.hidden_size,
        )
    for index in range(audio.num_hidden_layers):
        prefix = f"audio_tower.layers.{index}"
        expected.update({
            f"{prefix}.self_attn_layer_norm.weight": (audio.hidden_size,),
            f"{prefix}.final_layer_norm.weight": (audio.hidden_size,),
            f"{prefix}.self_attn.q_proj.weight": (audio.num_attention_heads * audio.head_dim, audio.hidden_size),
            f"{prefix}.self_attn.q_proj.bias": (audio.num_attention_heads * audio.head_dim,),
            f"{prefix}.self_attn.k_proj.weight": (audio.num_attention_heads * audio.head_dim, audio.hidden_size),
            f"{prefix}.self_attn.v_proj.weight": (audio.num_attention_heads * audio.head_dim, audio.hidden_size),
            f"{prefix}.self_attn.v_proj.bias": (audio.num_attention_heads * audio.head_dim,),
            f"{prefix}.self_attn.o_proj.weight": (audio.hidden_size, audio.num_attention_heads * audio.head_dim),
            f"{prefix}.self_attn.o_proj.bias": (audio.hidden_size,),
            f"{prefix}.mlp.gate_proj.weight": (audio.intermediate_size, audio.hidden_size),
            f"{prefix}.mlp.up_proj.weight": (audio.intermediate_size, audio.hidden_size),
            f"{prefix}.mlp.down_proj.weight": (audio.hidden_size, audio.intermediate_size),
            f"{prefix}.mlp.down_proj.bias": (audio.hidden_size,),
        })
    for index in range(text.num_hidden_layers):
        prefix = f"language_model.model.layers.{index}"
        expected.update({
            f"{prefix}.input_layernorm.weight": (text.hidden_size,),
            f"{prefix}.post_attention_layernorm.weight": (text.hidden_size,),
            f"{prefix}.self_attn.q_proj.weight": (text.num_attention_heads * text.head_dim, text.hidden_size),
            f"{prefix}.self_attn.q_proj.bias": (text.num_attention_heads * text.head_dim,),
            f"{prefix}.self_attn.k_proj.weight": (text.num_key_value_heads * text.head_dim, text.hidden_size),
            f"{prefix}.self_attn.k_proj.bias": (text.num_key_value_heads * text.head_dim,),
            f"{prefix}.self_attn.v_proj.weight": (text.num_key_value_heads * text.head_dim, text.hidden_size),
            f"{prefix}.self_attn.v_proj.bias": (text.num_key_value_heads * text.head_dim,),
            f"{prefix}.self_attn.o_proj.weight": (text.hidden_size, text.num_attention_heads * text.head_dim),
            f"{prefix}.mlp.gate_proj.weight": (text.intermediate_size, text.hidden_size),
            f"{prefix}.mlp.up_proj.weight": (text.intermediate_size, text.hidden_size),
            f"{prefix}.mlp.down_proj.weight": (text.hidden_size, text.intermediate_size),
            f"{prefix}.ada_rms_norm.linear1.weight": (32, text.hidden_size),
            f"{prefix}.ada_rms_norm.linear2.weight": (text.hidden_size, 32),
        })
    for index in range(len(config.semantic_vad_horizons_seconds)):
        expected[f"semantic_vad_heads.{index}.weight"] = (
            config.semantic_vad_num_classes, text.hidden_size,
        )
        expected[f"semantic_vad_heads.{index}.bias"] = (config.semantic_vad_num_classes,)
    return expected


def sanitize_state_dict(
    weights: dict[str, mx.array], config: Audio8Config
) -> tuple[dict[str, mx.array], tuple[str, ...]]:
    expected = expected_audio8_tensors(config)
    missing = sorted(set(expected) - set(weights))
    extra = sorted(set(weights) - set(expected))
    if missing or extra:
        raise ValueError(
            f"Audio8 checkpoint schema mismatch: {len(missing)} missing, "
            f"{len(extra)} unexpected; missing={missing[:3]}, unexpected={extra[:3]}"
        )
    sanitized: dict[str, mx.array] = {}
    transposed: list[str] = []
    for key in sorted(expected):
        value = weights[key]
        target_shape = expected[key]
        if key in CONV1D_WEIGHT_KEYS and tuple(value.shape) != target_shape:
            transposed_value = value.transpose(0, 2, 1) if value.ndim == 3 else value
            if tuple(transposed_value.shape) == target_shape:
                value = transposed_value
                transposed.append(key)
        if tuple(value.shape) != target_shape:
            raise ValueError(
                f"Audio8 tensor shape mismatch for {key}: "
                f"got {tuple(value.shape)}, expected {target_shape}"
            )
        sanitized[key] = value
    return sanitized, tuple(transposed)


def load_audio8_checkpoint(model_dir: str | Path) -> Audio8Checkpoint:
    model_dir = Path(model_dir)
    loaded = load_state_dict(model_dir)
    weights = dict(loaded.weights)
    source_files = list(loaded.files)
    vad_path = model_dir / "semantic_vad_heads.safetensors"
    if vad_path.exists() and vad_path not in source_files:
        source_files.append(vad_path)
        for key, value in mx.load(str(vad_path)).items():
            if key in weights:
                raise ValueError(f"Duplicate Audio8 tensor key: {key}")
            weights[key] = value
    config = Audio8Config.from_dir(model_dir)
    if get_audio8_quantization(config) is None:
        state_dict, transposed = sanitize_state_dict(weights, config)
    else:
        state_dict, transposed = weights, ()
    return Audio8Checkpoint(
        model_dir=model_dir,
        config=config,
        state_dict=state_dict,
        source_files=tuple(source_files),
        transposed_keys=transposed,
    )


def save_audio8_bf16_checkpoint(
    checkpoint: Audio8Checkpoint,
    output_dir: str | Path,
    *,
    copy_supporting_files: bool = True,
) -> Audio8ConversionReport:
    output_dir = Path(output_dir)
    if output_dir.resolve() == checkpoint.model_dir.resolve():
        raise ValueError("Audio8 conversion output must differ from the source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "model.safetensors"
    mx.eval(list(checkpoint.state_dict.values()))
    mx.save_safetensors(str(output_file), checkpoint.state_dict, metadata={"format": "mlx"})
    copied: list[Path] = []
    if copy_supporting_files:
        for name in SUPPORTING_FILES:
            source = checkpoint.model_dir / name
            if source.exists():
                destination = output_dir / name
                shutil.copy2(source, destination)
                copied.append(destination)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(checkpoint.config.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return Audio8ConversionReport(
        output_file=output_file,
        tensor_count=len(checkpoint.state_dict),
        transposed_keys=checkpoint.transposed_keys,
        copied_files=tuple(copied),
    )


def get_audio8_quantization(config: Audio8Config) -> Audio8QuantizationConfig | None:
    payload = config.extra.get("quantization")
    return None if payload is None else Audio8QuantizationConfig.from_dict(payload)


def quantize_audio8_model(
    model: nn.Module,
    quantization: Audio8QuantizationConfig,
    *,
    state_dict: dict[str, mx.array] | None = None,
) -> nn.Module:
    if quantization.bits != 8 or quantization.mode != "affine":
        raise ValueError("Audio8 currently supports only affine int8 quantization")
    if quantization.group_size not in (32, 64, 128):
        raise ValueError("Audio8 quantization group_size must be 32, 64, or 128")
    checkpoint_keys = set(state_dict) if state_dict is not None else None
    quantized_paths = None
    if checkpoint_keys is None:
        quantized_paths = {"language_model.model.embed_tokens"}
        if not model.config.text_config.tie_word_embeddings:
            quantized_paths.add("language_model.lm_head")
        blocks = (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for prefix, layers in (
            ("audio_tower.layers", model.config.audio_config.num_hidden_layers),
            ("language_model.model.layers", model.config.text_config.num_hidden_layers),
        ):
            for index in range(layers):
                for block, projections in blocks:
                    quantized_paths.update(
                        f"{prefix}.{index}.{block}.{projection}"
                        for projection in projections
                    )

    def should_quantize(path: str, module: Any) -> bool:
        if not isinstance(module, (nn.Linear, nn.Embedding)):
            return False
        if module.weight.shape[-1] % quantization.group_size:
            return False
        if checkpoint_keys is None:
            return path in quantized_paths
        return f"{path}.scales" in checkpoint_keys

    nn.quantize(
        model,
        group_size=quantization.group_size,
        bits=quantization.bits,
        mode=quantization.mode,
        class_predicate=should_quantize,
    )
    return model


def save_audio8_quantized_checkpoint(
    checkpoint: Audio8Checkpoint,
    output_dir: str | Path,
    quantization: Audio8QuantizationConfig,
    *,
    copy_supporting_files: bool = True,
) -> Audio8ConversionReport:
    if get_audio8_quantization(checkpoint.config) is not None:
        raise ValueError("Audio8 quantized conversion requires a BF16 source checkpoint")
    from .model import Audio8ASRModel

    model = Audio8ASRModel(checkpoint.config)
    load_checkpoint_into_model(model, checkpoint, strict=True)
    quantize_audio8_model(model, quantization)
    return save_audio8_model(
        model,
        output_dir,
        quantization=quantization,
        copy_supporting_from=checkpoint.model_dir if copy_supporting_files else None,
        transposed_keys=checkpoint.transposed_keys,
    )


def save_audio8_model(
    model: nn.Module,
    output_dir: str | Path,
    *,
    quantization: Audio8QuantizationConfig,
    copy_supporting_from: str | Path | None,
    transposed_keys: tuple[str, ...] = (),
) -> Audio8ConversionReport:
    output_dir = Path(output_dir)
    if copy_supporting_from is not None and output_dir.resolve() == Path(copy_supporting_from).resolve():
        raise ValueError("Audio8 conversion output must differ from the source directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        key: value
        for key, value in tree_flatten(model.parameters(), destination={}).items()
        if key not in GENERATED_MODEL_ONLY_KEYS
    }
    output_file = output_dir / "model.safetensors"
    mx.eval(list(weights.values()))
    mx.save_safetensors(
        str(output_file),
        weights,
        metadata={"format": "mlx", "quantization": "int8-affine"},
    )
    copied: list[Path] = []
    if copy_supporting_from is not None:
        for name in SUPPORTING_FILES:
            source = Path(copy_supporting_from) / name
            if source.exists():
                destination = output_dir / name
                shutil.copy2(source, destination)
                copied.append(destination)
    payload = model.config.to_dict()
    payload["quantization"] = quantization.to_dict()
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return Audio8ConversionReport(
        output_file=output_file,
        tensor_count=len(weights),
        transposed_keys=transposed_keys,
        copied_files=tuple(copied),
    )


def build_alignment_report(
    model_parameters: dict[str, mx.array], checkpoint_state: dict[str, mx.array]
) -> Audio8AlignmentReport:
    mismatches = []
    for key in sorted(set(model_parameters) & set(checkpoint_state)):
        model_shape = tuple(model_parameters[key].shape)
        checkpoint_shape = tuple(checkpoint_state[key].shape)
        if model_shape != checkpoint_shape:
            mismatches.append((key, model_shape, checkpoint_shape))
    return Audio8AlignmentReport(
        checkpoint_only=tuple(sorted(set(checkpoint_state) - set(model_parameters))),
        model_only=tuple(sorted(set(model_parameters) - set(checkpoint_state))),
        shape_mismatches=tuple(mismatches),
    )


def load_checkpoint_into_model(
    model: nn.Module, checkpoint: Audio8Checkpoint, *, strict: bool = True
) -> Audio8AlignmentReport:
    quantization = get_audio8_quantization(checkpoint.config)
    state_dict = checkpoint.state_dict
    if quantization is not None and checkpoint.config.text_config.tie_word_embeddings:
        state_dict = dict(state_dict)
        head_prefix = TIED_LM_HEAD_KEY.removesuffix(".weight")
        embedding_prefix = "language_model.model.embed_tokens"
        for suffix in ("weight", "scales", "biases"):
            head_key = f"{head_prefix}.{suffix}"
            if head_key not in state_dict:
                continue
            embedding_key = f"{embedding_prefix}.{suffix}"
            if embedding_key not in state_dict:
                raise ValueError("Audio8 tied language head requires its embedding tensors")
            if not mx.array_equal(state_dict[head_key], state_dict[embedding_key]).item():
                raise ValueError(
                    "Audio8 quantized tied weights differ; reconvert from the BF16 checkpoint"
                )
            del state_dict[head_key]
        quantize_audio8_model(model, quantization, state_dict=state_dict)
    elif quantization is not None:
        quantize_audio8_model(model, quantization, state_dict=state_dict)
    parameters = tree_flatten(model.parameters(), destination={})
    report = build_alignment_report(parameters, state_dict)
    allowed = set(GENERATED_MODEL_ONLY_KEYS)
    unexpected = tuple(key for key in report.model_only if key not in allowed)
    if strict and (report.checkpoint_only or unexpected or report.shape_mismatches):
        raise ValueError(
            f"Audio8 checkpoint alignment failed: {len(report.checkpoint_only)} checkpoint-only, "
            f"{len(unexpected)} unexpected model-only, {len(report.shape_mismatches)} shape mismatches"
        )
    model.load_weights(list(state_dict.items()), strict=strict and not report.model_only)
    return report
