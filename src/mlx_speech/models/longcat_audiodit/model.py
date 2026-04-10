"""Top-level LongCat AudioDiT runtime model."""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .config import LongCatAudioDiTConfig
from .sampling import _MomentumBuffer, apg_forward, odeint_euler


@dataclass(frozen=True)
class LongCatAudioDiTOutput:
    waveform: mx.array
    latent: mx.array


def _layer_norm(x: mx.array, *, eps: float = 1e-6) -> mx.array:
    mean = mx.mean(x.astype(mx.float32), axis=-1, keepdims=True)
    variance = mx.mean(mx.square(x.astype(mx.float32) - mean), axis=-1, keepdims=True)
    return ((x.astype(mx.float32) - mean) * mx.rsqrt(variance + eps)).astype(x.dtype)


def lens_to_mask(lengths: mx.array, *, length: int | None = None) -> mx.array:
    if length is None:
        length = int(mx.max(lengths).item())
    seq = mx.arange(length, dtype=lengths.dtype)
    return seq[None, :] < lengths[:, None]


class LongCatAudioDiTModel(nn.Module):
    def __init__(
        self, config: LongCatAudioDiTConfig, *, text_encoder, transformer, vae
    ) -> None:
        super().__init__()
        self.config = config
        self.text_encoder = text_encoder
        self.transformer = transformer
        self.vae = vae

    def encode_text(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        output = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        embedding = output.last_hidden_state
        if self.config.text_norm_feat:
            embedding = _layer_norm(embedding, eps=1e-6)
        if self.config.text_add_embed and output.hidden_states:
            first_hidden = output.hidden_states[0]
            if self.config.text_norm_feat:
                first_hidden = _layer_norm(first_hidden, eps=1e-6)
            embedding = embedding + first_hidden
        return embedding.astype(mx.float32)

    def encode_prompt_audio(self, prompt_audio: mx.array) -> tuple[mx.array, int]:
        full_hop = self.config.latent_hop
        off = 3
        wav = prompt_audio
        if wav.ndim == 2:
            wav = wav[:, None, :]
        if wav.shape[-1] % full_hop != 0:
            wav = mx.pad(
                wav, ((0, 0), (0, 0), (0, full_hop - (wav.shape[-1] % full_hop)))
            )
        wav = mx.pad(wav, ((0, 0), (0, 0), (0, full_hop * off)))
        latent = self.vae.encode(wav)
        if off:
            latent = latent[..., :-off]
        prompt_duration_frames = int(latent.shape[-1])
        return mx.transpose(latent, (0, 2, 1)), prompt_duration_frames

    def __call__(
        self,
        *,
        input_ids: mx.array | None = None,
        attention_mask: mx.array | None = None,
        text_embedding: mx.array | None = None,
        prompt_audio: mx.array | None = None,
        duration: int | None = None,
        steps: int = 16,
        cfg_strength: float = 4.0,
        guidance_method: str = "cfg",
        seed: int | None = None,
        initial_noise: mx.array | None = None,
    ) -> LongCatAudioDiTOutput:
        sr = self.config.sampling_rate
        full_hop = self.config.latent_hop
        max_duration_frames = int(self.config.max_wav_duration * sr // full_hop)
        repa_layer = self.config.repa_dit_layer

        if text_embedding is not None:
            text_condition = text_embedding.astype(mx.float32)
            if attention_mask is not None:
                text_condition_len = mx.sum(attention_mask, axis=1)
            else:
                text_condition_len = mx.full(
                    (text_condition.shape[0],), text_condition.shape[1], dtype=mx.int32
                )
        else:
            assert input_ids is not None and attention_mask is not None
            text_condition = self.encode_text(input_ids, attention_mask)
            text_condition_len = mx.sum(attention_mask, axis=1)

        batch = int(text_condition.shape[0])
        if prompt_audio is not None:
            prompt_latent, prompt_dur = self.encode_prompt_audio(prompt_audio)
        else:
            prompt_latent = mx.zeros(
                (batch, 0, self.config.latent_dim), dtype=mx.float32
            )
            prompt_dur = 0

        total_duration = (
            max_duration_frames
            if duration is None
            else min(int(duration), max_duration_frames)
        )
        duration_tensor = mx.full((batch,), total_duration, dtype=mx.int32)
        mask = lens_to_mask(duration_tensor)
        text_mask = lens_to_mask(text_condition_len, length=text_condition.shape[1])
        neg_text = mx.zeros_like(text_condition)
        neg_text_len = text_condition_len
        if prompt_audio is not None:
            gen_len = total_duration - prompt_dur
            latent_cond = mx.pad(prompt_latent, ((0, 0), (0, gen_len), (0, 0)))
            empty_latent_cond = mx.zeros_like(latent_cond)
        else:
            latent_cond = mx.zeros(
                (batch, total_duration, self.config.latent_dim), dtype=mx.float32
            )
            empty_latent_cond = latent_cond

        apg_buffer = (
            _MomentumBuffer(momentum=-0.3) if guidance_method == "apg" else None
        )

        def fn(t: mx.array, x: mx.array) -> mx.array:
            x_cond = x
            if prompt_dur:
                prompt_prefix = (prompt_noise * (1 - t)) + (
                    latent_cond[:, :prompt_dur] * t
                )
                x_cond = mx.concatenate([prompt_prefix, x[:, prompt_dur:]], axis=1)
            output = self.transformer(
                x=x_cond,
                text=text_condition,
                text_len=text_condition_len,
                time=t,
                mask=mask,
                cond_mask=text_mask,
                return_ith_layer=repa_layer,
                latent_cond=latent_cond,
            )
            pred = output["last_hidden_state"]
            if cfg_strength < 1e-5:
                return pred

            x_null = x
            if prompt_dur:
                x_null = mx.concatenate(
                    [mx.zeros_like(x[:, :prompt_dur]), x[:, prompt_dur:]],
                    axis=1,
                )
            null_output = self.transformer(
                x=x_null,
                text=neg_text,
                text_len=neg_text_len,
                time=t,
                mask=mask,
                cond_mask=text_mask,
                return_ith_layer=repa_layer,
                latent_cond=empty_latent_cond,
            )
            null_pred = null_output["last_hidden_state"]
            if guidance_method == "cfg":
                return pred + ((pred - null_pred) * cfg_strength)

            pred_sample = x[:, prompt_dur:] + ((1 - t) * pred[:, prompt_dur:])
            null_sample = x[:, prompt_dur:] + ((1 - t) * null_pred[:, prompt_dur:])
            out = apg_forward(
                pred_sample,
                null_sample,
                guidance_scale=cfg_strength,
                momentum_buffer=apg_buffer,
                eta=0.5,
                norm_threshold=0.0,
            )
            out = (out - x[:, prompt_dur:]) / (1 - t)
            if prompt_dur:
                return mx.pad(out, ((0, 0), (prompt_dur, 0), (0, 0)))
            return out

        if initial_noise is not None:
            expected_shape = (batch, total_duration, self.config.latent_dim)
            if tuple(int(dim) for dim in initial_noise.shape) != expected_shape:
                raise ValueError(
                    f"initial_noise must have shape {expected_shape}, got {tuple(int(dim) for dim in initial_noise.shape)}"
                )
            y0 = initial_noise.astype(mx.float32)
        else:
            if seed is not None:
                mx.random.seed(seed)
            y0 = mx.random.normal(
                (batch, total_duration, self.config.latent_dim), dtype=mx.float32
            )
        t = mx.linspace(0.0, 1.0, steps)
        prompt_noise = (
            y0[:, :prompt_dur]
            if prompt_dur
            else mx.zeros((batch, 0, self.config.latent_dim), dtype=mx.float32)
        )
        sampled = odeint_euler(fn, y0, t)[-1]
        pred_latent = sampled[:, prompt_dur:] if prompt_dur else sampled
        pred_latent = mx.transpose(pred_latent, (0, 2, 1)).astype(mx.float32)
        waveform = self.vae.decode(pred_latent).squeeze(1)
        return LongCatAudioDiTOutput(waveform=waveform, latent=pred_latent)
