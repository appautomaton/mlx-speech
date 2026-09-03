"""MLX autoregressive Qwen3/DiT core for FireRedTTS3 Base."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_speech.generation.fireredtts3 import (
    classifier_free_guidance,
    cosine_time_schedule,
    euler_flow_step,
)

from ..qwen3_asr.config import Qwen3ASRTextConfig
from ..qwen3_asr.text_decoder import Qwen3ASRTextKVCache, Qwen3ASRTextModel
from .config import FireRedTTS3Config
from .dit import DiT
from .patch_encoder import PatchEncoder, _linear


@dataclass(frozen=True)
class FireRedTTS3CoreConfig:
    redae_dim: int = 64
    num_history_patches: int = 2
    spk_in_dim: int = 512
    patch_size: int = 4
    patch_encoder_hidden_size: int = 1024
    patch_encoder_mlp_ratio: float = 4.0
    patch_encoder_depth: int = 8
    patch_encoder_num_heads: int = 16
    dit_mlp_ratio: float = 3.0
    dit_depth: int = 11
    dit_num_heads: int = 16
    dit_hidden_size: int = 1024
    qwen_hidden_size: int = 2048
    qwen_intermediate_size: int = 6144
    qwen_num_hidden_layers: int = 28
    qwen_num_attention_heads: int = 16
    qwen_num_key_value_heads: int = 8
    qwen_head_dim: int = 128
    qwen_vocab_size: int = 151936
    qwen_max_position_embeddings: int = 40960
    qwen_rope_theta: float = 1_000_000.0
    qwen_rms_norm_eps: float = 1e-6

    @classmethod
    def from_artifact(cls, artifact: FireRedTTS3Config) -> "FireRedTTS3CoreConfig":
        core = artifact.core
        qwen = artifact.qwen
        return cls(
            redae_dim=int(core["redae_dim"]),
            num_history_patches=int(core["num_history_patches"]),
            spk_in_dim=int(core["spk_in_dim"]),
            patch_size=int(core["patch_size"]),
            patch_encoder_hidden_size=int(core["patch_encoder_hidden_size"]),
            patch_encoder_mlp_ratio=float(core["patch_encoder_mlp_ratio"]),
            patch_encoder_depth=int(core["patch_encoder_depth"]),
            patch_encoder_num_heads=int(core["patch_encoder_num_heads"]),
            dit_mlp_ratio=float(core["dit_mlp_ratio"]),
            dit_depth=int(core["dit_depth"]),
            dit_num_heads=int(core["dit_num_heads"]),
            dit_hidden_size=int(core["dit_hidden_size"]),
            qwen_hidden_size=int(qwen["hidden_size"]),
            qwen_intermediate_size=int(qwen["intermediate_size"]),
            qwen_num_hidden_layers=int(qwen["num_hidden_layers"]),
            qwen_num_attention_heads=int(qwen["num_attention_heads"]),
            qwen_num_key_value_heads=int(qwen["num_key_value_heads"]),
            qwen_head_dim=int(qwen["head_dim"]),
            qwen_vocab_size=int(qwen["vocab_size"]),
            qwen_max_position_embeddings=int(qwen["max_position_embeddings"]),
            qwen_rope_theta=float(qwen["rope_theta"]),
            qwen_rms_norm_eps=float(qwen["rms_norm_eps"]),
        )

    def qwen_config(self) -> Qwen3ASRTextConfig:
        return Qwen3ASRTextConfig.from_dict(
            {
                "hidden_size": self.qwen_hidden_size,
                "intermediate_size": self.qwen_intermediate_size,
                "num_hidden_layers": self.qwen_num_hidden_layers,
                "num_attention_heads": self.qwen_num_attention_heads,
                "num_key_value_heads": self.qwen_num_key_value_heads,
                "head_dim": self.qwen_head_dim,
                "vocab_size": self.qwen_vocab_size,
                "max_position_embeddings": self.qwen_max_position_embeddings,
                "rms_norm_eps": self.qwen_rms_norm_eps,
                "rope_theta": self.qwen_rope_theta,
                "hidden_act": "silu",
                "attention_bias": False,
                "use_cache": True,
                "dtype": "bfloat16",
                "use_sliding_window": False,
            }
        )


@dataclass(frozen=True)
class CoreGenerationResult:
    latents: mx.array
    generated_patches: int
    stop_scores: tuple[float, ...]
    cache_length: int
    prompt_length: int


class FireRedTTS3Core(nn.Module):
    def __init__(self, config: FireRedTTS3CoreConfig):
        super().__init__()
        self.config = config
        qwen_config = config.qwen_config()
        self.backbone_llm = Qwen3ASRTextModel(qwen_config)
        self.spk_proj_llm = nn.Linear(config.spk_in_dim, config.qwen_hidden_size)
        self.spk_proj_dit = nn.Linear(config.spk_in_dim, config.spk_in_dim)
        self.patch_encoder = PatchEncoder(
            in_dim=config.redae_dim,
            out_dim=config.qwen_hidden_size,
            patch_size=config.patch_size,
            hidden_size=config.patch_encoder_hidden_size,
            mlp_ratio=config.patch_encoder_mlp_ratio,
            depth=config.patch_encoder_depth,
            num_heads=config.patch_encoder_num_heads,
        )
        self.dit_head = nn.Linear(config.qwen_hidden_size, config.dit_hidden_size)
        self.dit = DiT(
            in_channels=(
                config.redae_dim + config.spk_in_dim + config.dit_hidden_size
            ),
            out_channels=config.redae_dim,
            mlp_ratio=config.dit_mlp_ratio,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            hidden_size=config.dit_hidden_size,
        )
        self.stop_head = nn.Linear(config.qwen_hidden_size, 1)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "FireRedTTS3Core":
        root = Path(model_dir)
        artifact = FireRedTTS3Config.from_dir(root)
        model = cls(FireRedTTS3CoreConfig.from_artifact(artifact))
        weights = mx.load(root / artifact.files["core"])
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        return model

    @property
    def history_length(self) -> int:
        return self.config.num_history_patches * self.config.patch_size

    def _flow_one_step(
        self,
        *,
        history_latents: mx.array,
        backbone_condition: mx.array,
        speaker_condition: mx.array,
        time_schedule: mx.array,
        guidance_scale: float,
    ) -> mx.array:
        config = self.config
        noise = mx.random.normal((1, config.patch_size, config.redae_dim))
        state = mx.concatenate((history_latents, noise), axis=1)
        backbone = mx.repeat(backbone_condition, config.patch_size, axis=1)
        speaker = mx.broadcast_to(
            speaker_condition[:, None],
            (1, self.history_length + config.patch_size, config.spk_in_dim),
        )
        condition = mx.concatenate((backbone, speaker), axis=-1)
        for index in range(int(time_schedule.shape[0]) - 1):
            timestep = time_schedule[index : index + 1]
            delta = time_schedule[index + 1] - time_schedule[index]
            model_input = mx.concatenate((state, condition), axis=-1)
            if guidance_scale > 0:
                unconditioned = mx.concatenate((state, condition * 0), axis=-1)
                model_input = mx.concatenate((model_input, unconditioned), axis=0)
                timestep = mx.broadcast_to(timestep, (2,))
            velocity = self.dit(model_input, timestep)
            if guidance_scale > 0:
                conditional, unconditional = mx.split(velocity, 2, axis=0)
                velocity = classifier_free_guidance(
                    conditional,
                    unconditional,
                    guidance_scale,
                )
            current = euler_flow_step(
                state[:, -config.patch_size :],
                velocity[:, -config.patch_size :],
                delta,
            )
            state = mx.concatenate((history_latents, current), axis=1)
        return state[:, -config.patch_size :]

    def generate(
        self,
        *,
        speaker_embedding: mx.array,
        text_tokens: mx.array,
        prompt_latents: mx.array,
        flow_steps: int = 10,
        guidance_scale: float = 2.0,
        stop_threshold: float = 0.5,
        min_generated_patches: int | None = 6,
        max_generated_patches: int = 400,
        seed: int | None = 1234,
    ) -> CoreGenerationResult:
        self._validate_inputs(
            speaker_embedding=speaker_embedding,
            text_tokens=text_tokens,
            prompt_latents=prompt_latents,
            max_generated_patches=max_generated_patches,
        )
        if min_generated_patches is not None and min_generated_patches < 0:
            raise ValueError("min_generated_patches cannot be negative")
        if seed is not None:
            mx.random.seed(seed)

        text_embeddings = self.backbone_llm.embed_tokens(text_tokens)
        prompt_embeddings = self.patch_encoder(prompt_latents)
        speaker_llm = _linear(self.spk_proj_llm, speaker_embedding)[:, None]
        input_embeddings = mx.concatenate(
            (speaker_llm, text_embeddings, prompt_embeddings), axis=1
        )
        prompt_length = int(input_embeddings.shape[1])
        cache = Qwen3ASRTextKVCache.allocate(
            self.config.qwen_config(),
            batch_size=1,
            max_length=prompt_length + max_generated_patches,
            dtype=input_embeddings.dtype,
        )
        time_schedule = cosine_time_schedule(flow_steps)
        latents = mx.pad(
            prompt_latents,
            ((0, 0), (self.history_length, 0), (0, 0)),
        )
        speaker_dit = _linear(self.spk_proj_dit, speaker_embedding)
        backbone_condition = mx.zeros(
            (1, self.config.num_history_patches, self.config.qwen_hidden_size),
            dtype=input_embeddings.dtype,
        )

        stop_scores: list[float] = []
        generated_patches = 0
        for step_index in range(max_generated_patches):
            if step_index == 0:
                output = self.backbone_llm.prefill(
                    inputs_embeds=input_embeddings,
                    kv_cache=cache,
                )
            else:
                output = self.backbone_llm.decode_step(
                    inputs_embeds=input_embeddings,
                    kv_cache=cache,
                )
            hidden_states = output.last_hidden_state
            stop_score = float(
                mx.sigmoid(_linear(self.stop_head, hidden_states[:, -1])).item()
            )
            stop_scores.append(stop_score)
            can_stop = (
                min_generated_patches is None
                or step_index >= min_generated_patches
            )
            if stop_score >= stop_threshold and can_stop:
                break

            if step_index == 0:
                next_condition = hidden_states[:, -int(prompt_embeddings.shape[1]) :]
            else:
                next_condition = hidden_states[:, -1:]
            backbone_condition = mx.concatenate(
                (backbone_condition, next_condition), axis=1
            )
            dit_condition = _linear(
                self.dit_head,
                backbone_condition[:, -(self.config.num_history_patches + 1) :],
            )
            next_latents = self._flow_one_step(
                history_latents=latents[:, -self.history_length :],
                backbone_condition=dit_condition,
                speaker_condition=speaker_dit,
                time_schedule=time_schedule,
                guidance_scale=guidance_scale,
            )
            mx.eval(next_latents)
            input_embeddings = self.patch_encoder(next_latents)
            latents = mx.concatenate((latents, next_latents), axis=1)
            generated_patches += 1

        latents = latents[:, self.history_length :]
        return CoreGenerationResult(
            latents=latents,
            generated_patches=generated_patches,
            stop_scores=tuple(stop_scores),
            cache_length=cache.current_length,
            prompt_length=prompt_length,
        )

    def _validate_inputs(
        self,
        *,
        speaker_embedding: mx.array,
        text_tokens: mx.array,
        prompt_latents: mx.array,
        max_generated_patches: int,
    ) -> None:
        config = self.config
        if speaker_embedding.shape != (1, config.spk_in_dim):
            raise ValueError(
                f"speaker_embedding must have shape (1, {config.spk_in_dim})"
            )
        if text_tokens.ndim != 2 or int(text_tokens.shape[0]) != 1:
            raise ValueError("text_tokens must have shape (1, tokens)")
        if (
            prompt_latents.ndim != 3
            or int(prompt_latents.shape[0]) != 1
            or int(prompt_latents.shape[-1]) != config.redae_dim
        ):
            raise ValueError(
                f"prompt_latents must have shape (1, frames, {config.redae_dim})"
            )
        if int(prompt_latents.shape[1]) % config.patch_size:
            raise ValueError("prompt latent frames must be divisible by patch_size")
        if max_generated_patches <= 0:
            raise ValueError("max_generated_patches must be positive")
        if (
            int(text_tokens.shape[1])
            + int(prompt_latents.shape[1]) // config.patch_size
            + max_generated_patches
            + 1
            > config.qwen_max_position_embeddings
        ):
            raise ValueError("FireRedTTS3 generation exceeds the Qwen3 context window")


__all__ = [
    "CoreGenerationResult",
    "FireRedTTS3Core",
    "FireRedTTS3CoreConfig",
]
