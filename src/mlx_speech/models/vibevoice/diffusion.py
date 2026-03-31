"""Diffusion prediction head and DPM-Solver scheduler for VibeVoice Large."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from .config import VibeVoiceDiffusionConfig


# --------------------------------------------------------------------------- #
# Timestep Embedder
# --------------------------------------------------------------------------- #

class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding → MLP.

    Checkpoint keys: t_embedder.mlp.0.weight, t_embedder.mlp.2.weight
    To match the Sequential(Linear, SiLU, Linear) naming, we use a list.
    """

    def __init__(self, hidden_size: int, freq_embed_size: int = 256):
        super().__init__()
        self.freq_embed_size = freq_embed_size
        # Match checkpoint: mlp.0 and mlp.2 (SiLU at index 1 has no params)
        self.mlp = [
            nn.Linear(freq_embed_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        ]

    @staticmethod
    def _sinusoidal_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(
            -math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half
        )
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            emb = mx.concatenate([emb, mx.zeros_like(emb[:, :1])], axis=-1)
        return emb

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self._sinusoidal_embedding(t, self.freq_embed_size)
        x = self.mlp[0](t_freq)
        x = self.mlp[1](x)
        x = self.mlp[2](x)
        return x


# --------------------------------------------------------------------------- #
# SwiGLU FFN (used in diffusion head, different from GELU in Block1D)
# --------------------------------------------------------------------------- #

class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


# --------------------------------------------------------------------------- #
# adaLN modulation helper
# --------------------------------------------------------------------------- #

def _modulate(x: mx.array, shift: mx.array, scale: mx.array) -> mx.array:
    return x * (1 + scale) + shift


# --------------------------------------------------------------------------- #
# RMSNorm variants for diffusion head
# --------------------------------------------------------------------------- #

class DiffusionRMSNorm(nn.Module):
    """RMSNorm with optional learnable weight."""

    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x_f32 = x.astype(mx.float32)
        norm = x_f32 * mx.rsqrt(mx.mean(x_f32 ** 2, axis=-1, keepdims=True) + self.eps)
        if self.elementwise_affine:
            return (norm * self.weight).astype(x.dtype)
        return norm.astype(x.dtype)


# --------------------------------------------------------------------------- #
# Head Layers
# --------------------------------------------------------------------------- #

class HeadLayer(nn.Module):
    """One adaLN-modulated SwiGLU FFN block in the diffusion head.

    Checkpoint: adaLN_modulation.1.weight (Sequential index 1 = the Linear after SiLU)
    """

    def __init__(self, embed_dim: int, ffn_dim: int, cond_dim: int, norm_eps: float = 1e-5):
        super().__init__()
        self.ffn = SwiGLUFFN(embed_dim, ffn_dim)
        self.norm = DiffusionRMSNorm(embed_dim, eps=norm_eps)
        # adaLN_modulation: Sequential(SiLU, Linear) → checkpoint stores index 1
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * embed_dim, bias=False),
        ]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = self.adaLN_modulation[1](self.adaLN_modulation[0](c))
        shift, scale, gate = mx.split(mod, 3, axis=-1)
        x = x + gate * self.ffn(_modulate(self.norm(x), shift, scale))
        return x


class FinalLayer(nn.Module):
    """Final diffusion head layer (2-part modulation, no gate).

    norm_final has no learnable weight (elementwise_affine=False).
    """

    def __init__(self, hidden_size: int, output_size: int, cond_size: int, norm_eps: float = 1e-5):
        super().__init__()
        self.norm_final = DiffusionRMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = [
            nn.SiLU(),
            nn.Linear(cond_size, 2 * hidden_size, bias=False),
        ]

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        mod = self.adaLN_modulation[1](self.adaLN_modulation[0](c))
        shift, scale = mx.split(mod, 2, axis=-1)
        x = _modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# --------------------------------------------------------------------------- #
# Diffusion Head
# --------------------------------------------------------------------------- #

class VibeVoiceDiffusionHead(nn.Module):
    """Diffusion prediction head for VibeVoice.

    Predicts velocity (v-prediction) from noisy latent + LM condition.
    """

    def __init__(self, config: VibeVoiceDiffusionConfig):
        super().__init__()
        h = config.hidden_size
        latent = config.latent_size
        ffn_dim = int(h * config.head_ffn_ratio)
        eps = config.rms_norm_eps

        self.noisy_images_proj = nn.Linear(latent, h, bias=False)
        self.cond_proj = nn.Linear(h, h, bias=False)
        self.t_embedder = TimestepEmbedder(h)

        self.layers = [
            HeadLayer(h, ffn_dim, h, norm_eps=eps) for _ in range(config.head_layers)
        ]
        self.final_layer = FinalLayer(h, latent, h, norm_eps=eps)

    def __call__(
        self,
        noisy_latent: mx.array,
        timesteps: mx.array,
        condition: mx.array,
    ) -> mx.array:
        """Predict noise/velocity.

        Args:
            noisy_latent: (B, latent_size)
            timesteps: (B,) float timestep values
            condition: (B, hidden_size) from LM hidden state

        Returns:
            (B, latent_size) predicted velocity
        """
        x = self.noisy_images_proj(noisy_latent)
        t = self.t_embedder(timesteps)
        c = self.cond_proj(condition) + t

        for layer in self.layers:
            x = layer(x, c)

        return self.final_layer(x, c)


# --------------------------------------------------------------------------- #
# DPM-Solver Multistep Scheduler
# Ported from mlx-audio (Blaizzy/mlx-audio), proven working with VibeVoice.
# --------------------------------------------------------------------------- #

def _betas_for_alpha_bar(num_timesteps: int, max_beta: float = 0.999) -> list[float]:
    """Cosine beta schedule matching diffusers convention."""
    import math as _math

    def alpha_bar_fn(t: float) -> float:
        return _math.cos((t + 0.008) / 1.008 * _math.pi / 2) ** 2

    betas = []
    for i in range(num_timesteps):
        t1 = i / num_timesteps
        t2 = (i + 1) / num_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return betas


@dataclass
class SchedulerOutput:
    prev_sample: mx.array
    x0_pred: mx.array | None = None


class DPMSolverMultistepScheduler:
    """DPM-Solver++ scheduler for VibeVoice diffusion inference.

    Ported from mlx-audio — uses the sigma-parameterization and exponential
    update formulas that match the diffusers reference implementation.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        prediction_type: str = "v_prediction",
        solver_order: int = 2,
        lower_order_final: bool = True,
        final_sigmas_type: str = "zero",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.solver_order = solver_order
        self.lower_order_final = lower_order_final
        self.final_sigmas_type = final_sigmas_type

        # Cosine beta schedule
        betas_list = _betas_for_alpha_bar(num_train_timesteps)
        betas = np.array(betas_list, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)

        self.alpha_t = mx.array(np.sqrt(alphas_cumprod), dtype=mx.float32)
        self.sigma_t = mx.array(np.sqrt(1 - alphas_cumprod), dtype=mx.float32)
        self.lambda_t = mx.log(self.alpha_t) - mx.log(self.sigma_t)

        # State
        self.num_inference_steps: int | None = None
        self.timesteps: list[int] | None = None
        self.model_outputs: list[mx.array | None] = [None] * solver_order
        self.lower_order_nums: int = 0
        self._step_index: int | None = None

        self._cached_alpha_t: list[float] = []
        self._cached_sigma_t: list[float] = []
        self._cached_lambda: list[float] = []

    def reset(self) -> None:
        self.model_outputs = [None] * self.solver_order
        self.lower_order_nums = 0
        self._step_index = None

    def set_timesteps(self, num_inference_steps: int) -> None:
        self.num_inference_steps = num_inference_steps

        # Linspace from max to near-zero, matching diffusers convention
        timestep_values = []
        for i in range(num_inference_steps):
            t = (self.num_train_timesteps - 1) * (1.0 - i / num_inference_steps)
            timestep_values.append(int(round(t)))
        self.timesteps = timestep_values

        # Precompute alpha/sigma/lambda via sigma-parameterization
        alpha_t_np = np.array(self.alpha_t.tolist())

        self._cached_alpha_t = []
        self._cached_sigma_t = []
        self._cached_lambda = []

        for t in timestep_values:
            sigma = np.sqrt((1 - alpha_t_np[t] ** 2) / (alpha_t_np[t] ** 2))
            alpha_val = 1.0 / np.sqrt(sigma ** 2 + 1.0)
            sigma_val = sigma * alpha_val
            lambda_val = np.log(alpha_val) - np.log(sigma_val)
            self._cached_alpha_t.append(float(alpha_val))
            self._cached_sigma_t.append(float(sigma_val))
            self._cached_lambda.append(float(lambda_val))

        # Sentinel for final step
        self._cached_alpha_t.append(1.0)
        self._cached_sigma_t.append(0.0)
        self._cached_lambda.append(float("inf"))

        self.reset()

    def _convert_model_output(
        self, model_output: mx.array, sample: mx.array, step_idx: int,
    ) -> mx.array:
        alpha_t = self._cached_alpha_t[step_idx]
        sigma_t = self._cached_sigma_t[step_idx]
        if self.prediction_type == "v_prediction":
            return alpha_t * sample - sigma_t * model_output
        elif self.prediction_type == "epsilon":
            return (sample - sigma_t * model_output) / alpha_t
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")

    def _first_order_update(
        self, x0_pred: mx.array, sample: mx.array, step_idx: int,
    ) -> mx.array:
        alpha_t = self._cached_alpha_t[step_idx + 1]
        sigma_next = self._cached_sigma_t[step_idx + 1]
        sigma_curr = self._cached_sigma_t[step_idx]

        lambda_next = self._cached_lambda[step_idx + 1]
        lambda_curr = self._cached_lambda[step_idx]
        h = lambda_next - lambda_curr

        sigma_ratio = sigma_next / sigma_curr if sigma_curr > 0 else 0.0
        exp_neg_h = math.exp(-h) if not math.isinf(h) else 0.0

        return sigma_ratio * sample - alpha_t * (exp_neg_h - 1.0) * x0_pred

    def _second_order_update(
        self,
        x0_pred: mx.array,
        prev_x0: mx.array,
        sample: mx.array,
        step_idx: int,
    ) -> mx.array:
        alpha_t = self._cached_alpha_t[step_idx + 1]
        sigma_next = self._cached_sigma_t[step_idx + 1]
        sigma_curr = self._cached_sigma_t[step_idx]

        lambda_next = self._cached_lambda[step_idx + 1]
        lambda_curr = self._cached_lambda[step_idx]
        lambda_prev = self._cached_lambda[step_idx - 1] if step_idx > 0 else lambda_curr

        h = lambda_next - lambda_curr
        h0 = lambda_curr - lambda_prev
        r0 = h0 / h if h != 0 else 1.0

        D0 = x0_pred
        D1 = (1.0 / r0) * (x0_pred - prev_x0) if r0 != 0 else mx.zeros_like(x0_pred)

        sigma_ratio = sigma_next / sigma_curr if sigma_curr > 0 else 0.0
        exp_neg_h = math.exp(-h) if not math.isinf(h) else 0.0

        return (
            sigma_ratio * sample
            - alpha_t * (exp_neg_h - 1.0) * D0
            - 0.5 * alpha_t * (exp_neg_h - 1.0) * D1
        )

    def step(
        self,
        model_output: mx.array,
        timestep: int,
        sample: mx.array,
        prev_x0: mx.array | None = None,
    ) -> SchedulerOutput:
        if self._step_index is None:
            self._step_index = 0

        idx = self._step_index

        x0_pred = self._convert_model_output(model_output, sample, idx)

        # Shift model output history
        for i in range(self.solver_order - 1, 0, -1):
            self.model_outputs[i] = self.model_outputs[i - 1]
        self.model_outputs[0] = x0_pred

        # Determine order
        lower_order_final_flag = (
            idx == self.num_inference_steps - 1
        ) and (
            (self.lower_order_final and self.num_inference_steps < 15)
            or self.final_sigmas_type == "zero"
        )

        if self.lower_order_nums < 1 or lower_order_final_flag:
            order = 1
        elif self.solver_order == 2 or self.lower_order_nums < 2:
            order = 2
        else:
            order = self.solver_order

        # Update
        if order == 1:
            prev_sample = self._first_order_update(x0_pred, sample, idx)
        elif order == 2:
            prev_x0_for_update = prev_x0 or self.model_outputs[1]
            if prev_x0_for_update is not None:
                prev_sample = self._second_order_update(
                    x0_pred, prev_x0_for_update, sample, idx,
                )
            else:
                prev_sample = self._first_order_update(x0_pred, sample, idx)
        else:
            prev_sample = self._first_order_update(x0_pred, sample, idx)

        if self.lower_order_nums < self.solver_order - 1:
            self.lower_order_nums += 1

        self._step_index += 1
        return SchedulerOutput(prev_sample=prev_sample, x0_pred=x0_pred)
