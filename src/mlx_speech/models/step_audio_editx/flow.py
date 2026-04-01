"""Non-stream CosyVoice flow conditioning helpers for Step-Audio-EditX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from ...checkpoints import load_torch_archive_state_dict
from .frontend import resolve_step_audio_cosyvoice_dir


def _flatten_token_sequence(tokens: Sequence[int] | np.ndarray) -> list[int]:
    array = np.asarray(tokens, dtype=np.int64)
    if array.ndim == 0:
        return [int(array)]
    if array.ndim == 1:
        return [int(value) for value in array.tolist()]
    if array.ndim == 2 and 1 in array.shape:
        return [int(value) for value in array.reshape(-1).tolist()]
    raise ValueError(f"Expected a flat mixed token sequence, got shape {array.shape}.")


def reshape_mixed_audio_tokens(
    mixed_tokens: Sequence[int] | np.ndarray,
    *,
    vq02_pad_token: int = 1024,
    vq06_prompt_offset: int = 1024,
    vq06_vocoder_base: int = 1025,
) -> np.ndarray:
    """Match the shipped CosyVoice `_reshape` helper exactly."""

    mixed = _flatten_token_sequence(mixed_tokens)
    remainder = len(mixed) % 5
    if remainder:
        pad_len = 5 - remainder
        mixed = mixed + [0, 0, 0, 1024, 1024, 1024][-pad_len:]

    num_groups = len(mixed) // 5
    vq02: list[int] = []
    vq06: list[int] = []
    for group_index in range(num_groups):
        start = group_index * 5
        vq02.extend(mixed[start : start + 2])
        vq02.append(int(vq02_pad_token))
        vq06.extend(mixed[start + 2 : start + 5])

    return np.stack(
        [
            np.asarray(vq02, dtype=np.int64),
            np.asarray(vq06, dtype=np.int64) - int(vq06_prompt_offset) + int(vq06_vocoder_base),
        ],
        axis=1,
    )


def interpolate_prompt_features(
    prompt_feat: np.ndarray,
    *,
    target_length: int,
) -> np.ndarray:
    """Nearest-neighbor time interpolation matching the shipped non-stream path."""

    feat = np.asarray(prompt_feat, dtype=np.float32)
    if feat.ndim != 3:
        raise ValueError(
            f"Expected prompt features with shape (batch, time, channels), got {feat.shape}."
        )
    if target_length <= 0:
        raise ValueError(f"Expected positive target length, got {target_length}.")

    input_length = int(feat.shape[1])
    if input_length == target_length:
        return feat.copy()
    indices = np.floor(
        np.arange(target_length, dtype=np.float32) * float(input_length) / float(target_length)
    ).astype(np.int64)
    indices = np.clip(indices, 0, max(input_length - 1, 0))
    return feat[:, indices, :].astype(np.float32, copy=False)


class StepAudioDualCodebookEmbedding(nn.Module):
    def __init__(self, vocab_size: int, input_size: int):
        super().__init__()
        if input_size % 2 != 0:
            raise ValueError(f"Expected even input_size for dual codebooks, got {input_size}.")
        self.embedding = nn.Embedding(vocab_size, input_size // 2)

    def __call__(self, tokens: mx.array) -> mx.array:
        embed1 = self.embedding(tokens[..., 0])
        embed2 = self.embedding(tokens[..., 1])
        return mx.concatenate([embed1, embed2], axis=-1)


@dataclass(frozen=True)
class StepAudioFlowConditioningConfig:
    vocab_size: int
    input_size: int
    output_size: int
    spk_embed_dim: int
    vq02_pad_token: int = 1024
    vq06_prompt_offset: int = 1024
    vq06_vocoder_base: int = 1025
    prompt_mel_upsample: int = 2

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, mx.array]) -> "StepAudioFlowConditioningConfig":
        embedding_weight = state_dict["input_embedding.embedding.weight"]
        spk_weight = state_dict["spk_embed_affine_layer.weight"]
        return cls(
            vocab_size=int(embedding_weight.shape[0]),
            input_size=int(embedding_weight.shape[1] * 2),
            output_size=int(spk_weight.shape[0]),
            spk_embed_dim=int(spk_weight.shape[1]),
        )


@dataclass(frozen=True)
class StepAudioFlowConditioningCheckpoint:
    model_dir: Path
    config: StepAudioFlowConditioningConfig
    state_dict: dict[str, mx.array]


@dataclass(frozen=True)
class StepAudioFlowConditioningAlignmentReport:
    missing_in_model: tuple[str, ...]
    missing_in_checkpoint: tuple[str, ...]
    shape_mismatches: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]

    @property
    def is_exact_match(self) -> bool:
        return (
            not self.missing_in_model
            and not self.missing_in_checkpoint
            and not self.shape_mismatches
        )


@dataclass(frozen=True)
class PreparedStepAudioNonStreamInputs:
    token_dual: np.ndarray
    prompt_token_dual: np.ndarray
    concatenated_token_dual: np.ndarray
    concatenated_token_length: np.ndarray
    embedded_tokens: np.ndarray
    prompt_feat_aligned: np.ndarray
    normalized_speaker_embedding: np.ndarray
    projected_speaker_embedding: np.ndarray


@dataclass(frozen=True)
class LoadedStepAudioFlowConditioner:
    model_dir: Path
    config: StepAudioFlowConditioningConfig
    checkpoint: StepAudioFlowConditioningCheckpoint
    model: "StepAudioFlowConditioner"
    alignment_report: StepAudioFlowConditioningAlignmentReport


class StepAudioFlowConditioner(nn.Module):
    def __init__(self, config: StepAudioFlowConditioningConfig):
        super().__init__()
        self.config = config
        self.input_embedding = StepAudioDualCodebookEmbedding(
            vocab_size=config.vocab_size,
            input_size=config.input_size,
        )
        self.spk_embed_affine_layer = nn.Linear(
            config.spk_embed_dim,
            config.output_size,
            bias=True,
        )

    def normalize_speaker_embedding(self, embedding: np.ndarray) -> np.ndarray:
        speaker_embedding = np.asarray(embedding, dtype=np.float32)
        if speaker_embedding.ndim == 1:
            speaker_embedding = speaker_embedding[None, :]
        if speaker_embedding.ndim != 2 or speaker_embedding.shape[1] != self.config.spk_embed_dim:
            raise ValueError(
                f"Expected speaker embedding shape (*, {self.config.spk_embed_dim}), "
                f"got {speaker_embedding.shape}."
            )
        norm = np.linalg.norm(speaker_embedding, axis=1, keepdims=True)
        norm = np.maximum(norm, 1e-12)
        return (speaker_embedding / norm).astype(np.float32)

    def project_speaker_embedding(self, embedding: np.ndarray) -> np.ndarray:
        normalized = self.normalize_speaker_embedding(embedding)
        projected = self.spk_embed_affine_layer(mx.array(normalized, dtype=mx.float32))
        return np.asarray(projected, dtype=np.float32)

    def embed_dual_tokens(self, tokens: np.ndarray) -> np.ndarray:
        dual = np.asarray(tokens, dtype=np.int64)
        if dual.ndim != 3 or dual.shape[-1] != 2:
            raise ValueError(
                f"Expected dual tokens with shape (batch, time, 2), got {dual.shape}."
            )
        embedded = self.input_embedding(mx.array(np.maximum(dual, 0), dtype=mx.int64))
        return np.asarray(embedded, dtype=np.float32)

    def prepare_nonstream_inputs(
        self,
        token: Sequence[int] | np.ndarray,
        prompt_token: Sequence[int] | np.ndarray,
        prompt_feat: np.ndarray,
        speaker_embedding: np.ndarray,
    ) -> PreparedStepAudioNonStreamInputs:
        token_dual = reshape_mixed_audio_tokens(
            token,
            vq02_pad_token=self.config.vq02_pad_token,
            vq06_prompt_offset=self.config.vq06_prompt_offset,
            vq06_vocoder_base=self.config.vq06_vocoder_base,
        )[None, :, :]
        prompt_token_dual = reshape_mixed_audio_tokens(
            prompt_token,
            vq02_pad_token=self.config.vq02_pad_token,
            vq06_prompt_offset=self.config.vq06_prompt_offset,
            vq06_vocoder_base=self.config.vq06_vocoder_base,
        )[None, :, :]

        prompt_feat_aligned = interpolate_prompt_features(
            prompt_feat,
            target_length=int(prompt_token_dual.shape[1]) * self.config.prompt_mel_upsample,
        )
        normalized_speaker_embedding = self.normalize_speaker_embedding(speaker_embedding)
        projected_speaker_embedding = self.project_speaker_embedding(
            normalized_speaker_embedding
        )

        concatenated_token_dual = np.concatenate([prompt_token_dual, token_dual], axis=1)
        concatenated_token_length = np.asarray(
            [concatenated_token_dual.shape[1]],
            dtype=np.int64,
        )
        embedded_tokens = self.embed_dual_tokens(concatenated_token_dual)

        return PreparedStepAudioNonStreamInputs(
            token_dual=token_dual.astype(np.int64, copy=False),
            prompt_token_dual=prompt_token_dual.astype(np.int64, copy=False),
            concatenated_token_dual=concatenated_token_dual.astype(np.int64, copy=False),
            concatenated_token_length=concatenated_token_length,
            embedded_tokens=embedded_tokens,
            prompt_feat_aligned=prompt_feat_aligned.astype(np.float32, copy=False),
            normalized_speaker_embedding=normalized_speaker_embedding,
            projected_speaker_embedding=projected_speaker_embedding,
        )


def sanitize_step_audio_flow_conditioning_state_dict(
    state_dict: dict[str, mx.array],
) -> tuple[StepAudioFlowConditioningConfig, dict[str, mx.array]]:
    config = StepAudioFlowConditioningConfig.from_state_dict(state_dict)
    selected = {
        "input_embedding.embedding.weight": state_dict["input_embedding.embedding.weight"],
        "spk_embed_affine_layer.weight": state_dict["spk_embed_affine_layer.weight"],
        "spk_embed_affine_layer.bias": state_dict["spk_embed_affine_layer.bias"],
    }
    return config, selected


def load_step_audio_flow_conditioning_checkpoint(
    model_dir: str | Path,
) -> StepAudioFlowConditioningCheckpoint:
    resolved_model_dir = resolve_step_audio_cosyvoice_dir(model_dir)
    archive = load_torch_archive_state_dict(resolved_model_dir / "flow.pt")
    config, state_dict = sanitize_step_audio_flow_conditioning_state_dict(archive.weights)
    return StepAudioFlowConditioningCheckpoint(
        model_dir=resolved_model_dir,
        config=config,
        state_dict=state_dict,
    )


def validate_step_audio_flow_conditioning_checkpoint_against_model(
    model: StepAudioFlowConditioner,
    checkpoint: StepAudioFlowConditioningCheckpoint,
) -> StepAudioFlowConditioningAlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        checkpoint_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return StepAudioFlowConditioningAlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_step_audio_flow_conditioner(
    model_dir: str | Path,
    *,
    strict: bool = True,
) -> LoadedStepAudioFlowConditioner:
    checkpoint = load_step_audio_flow_conditioning_checkpoint(model_dir)
    model = StepAudioFlowConditioner(checkpoint.config)
    report = validate_step_audio_flow_conditioning_checkpoint_against_model(
        model,
        checkpoint,
    )
    if strict and not report.is_exact_match:
        raise ValueError(
            "Step-Audio flow-conditioning checkpoint/model alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only keys, "
            f"{len(report.missing_in_checkpoint)} model-only keys, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    return LoadedStepAudioFlowConditioner(
        model_dir=checkpoint.model_dir,
        config=checkpoint.config,
        checkpoint=checkpoint,
        model=model,
        alignment_report=report,
    )


__all__ = [
    "LoadedStepAudioFlowConditioner",
    "PreparedStepAudioNonStreamInputs",
    "StepAudioDualCodebookEmbedding",
    "StepAudioFlowConditioner",
    "StepAudioFlowConditioningAlignmentReport",
    "StepAudioFlowConditioningCheckpoint",
    "StepAudioFlowConditioningConfig",
    "interpolate_prompt_features",
    "load_step_audio_flow_conditioner",
    "load_step_audio_flow_conditioning_checkpoint",
    "reshape_mixed_audio_tokens",
    "sanitize_step_audio_flow_conditioning_state_dict",
    "validate_step_audio_flow_conditioning_checkpoint_against_model",
]
