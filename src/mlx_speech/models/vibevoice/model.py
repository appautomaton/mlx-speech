"""Top-level VibeVoice Large model for conditional generation."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .acoustic import VibeVoiceAcousticTokenizer, VibeVoiceConvCache, VibeVoiceSemanticTokenizer
from .config import VibeVoiceConfig
from .connector import SpeechConnector
from .diffusion import DPMSolverMultistepScheduler, VibeVoiceDiffusionHead
from .qwen2 import Qwen2Model


class VibeVoiceModel(nn.Module):
    """Inner model container matching checkpoint ``model.*`` keys."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.language_model = Qwen2Model(config.language_config)
        self.acoustic_tokenizer = VibeVoiceAcousticTokenizer(config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibeVoiceSemanticTokenizer(config.semantic_tokenizer_config)
        self.prediction_head = VibeVoiceDiffusionHead(config.diffusion_config)
        self.acoustic_connector = SpeechConnector(
            config.acoustic_vae_dim, config.language_config.hidden_size,
        )
        self.semantic_connector = SpeechConnector(
            config.semantic_vae_dim, config.language_config.hidden_size,
        )

        # Scalar buffers loaded from checkpoint
        self.speech_scaling_factor = mx.array(1.0)
        self.speech_bias_factor = mx.array(0.0)

        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_config.ddpm_num_steps,
            beta_schedule=config.diffusion_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_config.prediction_type,
        )


class VibeVoiceForConditionalGeneration(nn.Module):
    """Top-level model matching checkpoint key structure."""

    def __init__(self, config: VibeVoiceConfig):
        super().__init__()
        self.config = config
        self.model = VibeVoiceModel(config)
        self.lm_head = nn.Linear(
            config.language_config.hidden_size,
            config.language_config.vocab_size,
            bias=False,
        )
        self.ddpm_inference_steps = config.diffusion_config.ddpm_num_inference_steps

    # -- Convenience properties ------------------------------------------------

    @property
    def language_model(self) -> Qwen2Model:
        return self.model.language_model

    @property
    def embed_tokens(self) -> nn.Embedding:
        return self.model.language_model.embed_tokens

    # -- Encode reference audio -----------------------------------------------

    def encode_reference_audio(self, audio: mx.array) -> mx.array:
        """Encode reference waveform → scaled+connected acoustic embeddings.

        Args:
            audio: (B, 1, T) raw waveform at 24 kHz

        Returns:
            connected: (B, T_frames, hidden_size) embeddings for splicing
        """
        mean = self.model.acoustic_tokenizer.encode(audio)  # (B, T_frames, vae_dim)
        sampled = self.model.acoustic_tokenizer.sample(mean)
        scaled = (sampled + self.model.speech_bias_factor) * self.model.speech_scaling_factor
        return self.model.acoustic_connector(scaled)

    # -- LM forward -----------------------------------------------------------

    def lm_forward(
        self,
        *,
        inputs_embeds: mx.array,
        cache: list[tuple[mx.array, mx.array]] | None = None,
    ) -> tuple[mx.array, mx.array, list[tuple[mx.array, mx.array]]]:
        """Run one LM step.

        Returns:
            logits: (B, L, vocab_size)
            hidden: (B, L, hidden_size)
            new_cache: updated KV cache
        """
        out = self.model.language_model(inputs_embeds=inputs_embeds, cache=cache)
        logits = self.lm_head(out.last_hidden_state)
        return logits, out.last_hidden_state, out.cache

    # -- Diffusion sampling ---------------------------------------------------

    def sample_speech_tokens(
        self,
        condition: mx.array,
        neg_condition: mx.array,
        *,
        cfg_scale: float = 3.0,
        num_steps: int | None = None,
    ) -> mx.array:
        """Run diffusion denoising with CFG to produce one acoustic latent frame.

        Faithfully matches the torch reference in modeling_vibevoice_inference.py:
        - Both positive and negative speech halves are tracked together
        - Only the positive half is passed through the head each step
        - CFG is applied, then duplicated back to update both halves via scheduler

        Args:
            condition: (B, hidden_size) positive conditioning from LM
            neg_condition: (B, hidden_size) negative conditioning
            cfg_scale: classifier-free guidance scale
            num_steps: override inference steps

        Returns:
            speech_latent: (B, latent_size) denoised acoustic latent
        """
        steps = num_steps or self.ddpm_inference_steps
        scheduler = self.model.noise_scheduler

        scheduler.reset()
        scheduler.set_timesteps(steps)

        # Concatenate conditions: [positive, negative] along batch dim
        cond_combined = mx.concatenate(
            [condition.astype(mx.float32), neg_condition.astype(mx.float32)], axis=0,
        )  # (2*B, H)

        B = condition.shape[0]
        latent_dim = self.config.diffusion_config.latent_size

        # Initialize speech noise for BOTH halves (matches torch reference)
        speech = mx.random.normal(
            (cond_combined.shape[0], latent_dim), dtype=mx.float32,
        )  # (2*B, 64)

        for t_val in scheduler.timesteps:
            # Take positive half only, duplicate for batched head forward
            half = speech[:B]
            combined = mx.concatenate([half, half], axis=0)  # (2*B, 64)

            t = mx.full((combined.shape[0],), float(t_val), dtype=mx.float32)
            eps = self.model.prediction_head(combined, t, condition=cond_combined)

            # CFG: split into conditional and unconditional
            cond_eps = eps[:B]
            uncond_eps = eps[B:]
            guided = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

            # Duplicate guided eps back to full batch for scheduler
            full_eps = mx.concatenate([guided, guided], axis=0)  # (2*B, 64)

            # Scheduler updates the full (2*B) speech tensor
            out = scheduler.step(full_eps, t_val, speech)
            speech = out.prev_sample  # (2*B, 64)

        return speech[:B]

    # -- Decode latent to audio -----------------------------------------------

    def decode_latent_to_audio(
        self,
        latent: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        """Decode acoustic latent → waveform.

        Args:
            latent: (B, latent_size) scaled speech latent
            cache: optional streaming conv cache for per-frame decoding

        Returns:
            audio: (B, 1, T_audio) waveform chunk
        """
        # Invert scaling: raw = latent / scale - bias
        raw = latent / self.model.speech_scaling_factor - self.model.speech_bias_factor
        # Reshape for decoder: (B, vae_dim, 1) — one latent frame, vae_dim channels
        raw = mx.expand_dims(raw, axis=2)  # (B, vae_dim, 1)
        return self.model.acoustic_tokenizer.decode(raw, cache=cache)

    # -- Semantic feedback ----------------------------------------------------

    def encode_semantic(
        self,
        audio: mx.array,
        *,
        cache: VibeVoiceConvCache | None = None,
    ) -> mx.array:
        """Encode audio chunk → semantic latent for feedback.

        Args:
            audio: (B, 1, T_audio) waveform chunk
            cache: optional streaming conv cache for per-frame encoding

        Returns:
            semantic: (B, T_frames, semantic_vae_dim)
        """
        return self.model.semantic_tokenizer.encode(audio, cache=cache)
