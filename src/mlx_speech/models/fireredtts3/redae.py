"""MLX implementation of the FireRedTTS3 RedAE waveform autoencoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..qwen3_asr.config import Qwen3ASRTextConfig
from ..qwen3_asr.text_decoder import Qwen3ASRTextModel
from .config import FireRedTTS3Config


def _qwen_config(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    max_position_embeddings: int,
    vocab_size: int,
    rope_theta: float,
    sliding_window: int | None = None,
    max_window_layers: int = 0,
) -> Qwen3ASRTextConfig:
    return Qwen3ASRTextConfig.from_dict(
        {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "max_position_embeddings": max_position_embeddings,
            "rms_norm_eps": 1e-6,
            "rope_theta": rope_theta,
            "hidden_act": "silu",
            "attention_bias": False,
            "use_cache": False,
            "dtype": "bfloat16",
            "use_sliding_window": sliding_window is not None,
            "sliding_window": sliding_window,
            "max_window_layers": max_window_layers,
        }
    )


@dataclass(frozen=True)
class RedAEConfig:
    audio_patch_size: int = 480
    audio_sample_rate: int = 24_000
    bottleneck_dim: int = 64
    head_dim: int = 128
    enc_hidden_size: int = 896
    enc_intermediate_size: int = 3584
    enc_num_hidden_layers: int = 18
    enc_max_position_embeddings: int = 32768
    enc_num_attention_heads: int = 14
    enc_num_key_value_heads: int = 2
    enc_sliding_window: int = 64
    enc_max_window_layers: int = 0
    enc_extra_downsample_rate: int = 2
    enc_downsample_num_hidden_layers: int = 4
    dec_hidden_size: int = 896
    dec_intermediate_size: int = 3584
    dec_num_hidden_layers: int = 18
    dec_max_position_embeddings: int = 32768
    dec_num_attention_heads: int = 14
    dec_num_key_value_heads: int = 2
    dec_sliding_window: int = 64
    dec_max_window_layers: int = 0
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0

    @classmethod
    def from_artifact(cls, artifact: FireRedTTS3Config) -> "RedAEConfig":
        source = artifact.redae
        qwen = artifact.qwen
        return cls(
            audio_patch_size=int(source["audio_patch_size"]),
            audio_sample_rate=int(source["audio_sample_rate"]),
            bottleneck_dim=int(source["bottleneck_dim"]),
            head_dim=int(qwen["head_dim"]),
            enc_hidden_size=int(source["enc_hidden_size"]),
            enc_intermediate_size=int(source["enc_intermediate_size"]),
            enc_num_hidden_layers=int(source["enc_num_hidden_layers"]),
            enc_max_position_embeddings=int(source["enc_max_position_embeddings"]),
            enc_num_attention_heads=int(source["enc_num_attention_heads"]),
            enc_num_key_value_heads=int(source["enc_num_key_value_heads"]),
            enc_sliding_window=int(source["enc_sliding_window"]),
            enc_max_window_layers=int(source["enc_max_window_layers"]),
            enc_extra_downsample_rate=int(source["enc_extra_downsample_rate"]),
            enc_downsample_num_hidden_layers=int(
                source["enc_downsample_num_hidden_layers"]
            ),
            dec_hidden_size=int(source["dec_hidden_size"]),
            dec_intermediate_size=int(source["dec_intermediate_size"]),
            dec_num_hidden_layers=int(source["dec_num_hidden_layers"]),
            dec_max_position_embeddings=int(source["dec_max_position_embeddings"]),
            dec_num_attention_heads=int(source["dec_num_attention_heads"]),
            dec_num_key_value_heads=int(source["dec_num_key_value_heads"]),
            dec_sliding_window=int(source["dec_sliding_window"]),
            dec_max_window_layers=int(source["dec_max_window_layers"]),
            vocab_size=int(qwen["vocab_size"]),
            rope_theta=float(qwen["rope_theta"]),
        )


class Qwen3CLSDownsample(nn.Module):
    def __init__(self, config: RedAEConfig):
        super().__init__()
        hidden = config.enc_hidden_size
        self.downsample_rate = config.enc_extra_downsample_rate
        self.cls_tok = mx.ones((1, 1, hidden))
        self.qwen3 = Qwen3ASRTextModel(
            _qwen_config(
                hidden_size=hidden,
                intermediate_size=config.enc_intermediate_size,
                num_hidden_layers=config.enc_downsample_num_hidden_layers,
                num_attention_heads=config.enc_num_attention_heads,
                num_key_value_heads=config.enc_num_key_value_heads,
                head_dim=config.head_dim,
                max_position_embeddings=config.enc_max_position_embeddings,
                vocab_size=config.vocab_size,
                rope_theta=config.rope_theta,
            )
        )

    def __call__(self, value: mx.array) -> mx.array:
        if int(value.shape[1]) % self.downsample_rate:
            raise ValueError(
                "RedAE encoder frames must be divisible by the CLS downsample rate"
            )
        batch, frames, hidden = value.shape
        value = value.reshape(
            batch * frames // self.downsample_rate,
            self.downsample_rate,
            hidden,
        )
        cls = mx.broadcast_to(self.cls_tok, (value.shape[0], 1, hidden))
        value = mx.concatenate((value, cls), axis=1)
        value = self.qwen3(inputs_embeds=value).last_hidden_state[:, -1]
        return value.reshape(batch, frames // self.downsample_rate, hidden)


class RedAEAudioEncoder(nn.Module):
    def __init__(self, config: RedAEConfig):
        super().__init__()
        self.config = config
        self.in_proj = [
            nn.Linear(config.audio_patch_size, config.enc_hidden_size),
            nn.Linear(config.enc_hidden_size, config.enc_hidden_size),
        ]
        self.qwen3 = Qwen3ASRTextModel(
            _qwen_config(
                hidden_size=config.enc_hidden_size,
                intermediate_size=config.enc_intermediate_size,
                num_hidden_layers=config.enc_num_hidden_layers,
                num_attention_heads=config.enc_num_attention_heads,
                num_key_value_heads=config.enc_num_key_value_heads,
                head_dim=config.head_dim,
                max_position_embeddings=config.enc_max_position_embeddings,
                vocab_size=config.vocab_size,
                rope_theta=config.rope_theta,
                sliding_window=config.enc_sliding_window,
                max_window_layers=config.enc_max_window_layers,
            )
        )
        self.downsample = Qwen3CLSDownsample(config)
        self.out_proj = nn.Linear(config.enc_hidden_size, config.bottleneck_dim)

    @property
    def downsample_rate(self) -> int:
        return self.config.audio_patch_size * self.config.enc_extra_downsample_rate

    def __call__(self, audio: mx.array) -> mx.array:
        if audio.ndim != 2:
            raise ValueError(f"RedAE expects audio with shape (batch, samples), got {audio.shape}")
        if int(audio.shape[1]) % self.downsample_rate:
            raise ValueError(
                f"RedAE audio length must be divisible by {self.downsample_rate}"
            )
        batch = int(audio.shape[0])
        value = audio.reshape(batch, -1, self.config.audio_patch_size)
        value = self.in_proj[1](self.in_proj[0](value))
        value = self.qwen3(inputs_embeds=value).last_hidden_state
        value = self.downsample(value)
        return self.out_proj(value)


def _overlap_add(frames: mx.array, hop: int) -> mx.array:
    batch, frame_count, frame_size = frames.shape
    output_size = (int(frame_count) - 1) * hop + int(frame_size)
    output = mx.zeros((batch, output_size), dtype=frames.dtype)
    columns = mx.arange(frame_size)
    for index in range(int(frame_count)):
        output[:, index * hop + columns] += frames[:, index]
    return output


class ISTFT(nn.Module):
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        periodic_hann = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)
        self.window = mx.array(periodic_hann)

    def __call__(self, spectrum: mx.array) -> mx.array:
        if spectrum.ndim != 3:
            raise ValueError("RedAE ISTFT expects (batch, frames, frequencies)")
        frames = mx.fft.irfft(spectrum, n=self.n_fft, axis=-1).real
        window = self.window.astype(mx.float32)
        frames = frames.astype(mx.float32) * window[None, None]
        output = _overlap_add(frames, self.hop_length)
        envelope_frames = mx.broadcast_to(
            (window * window)[None, None],
            (1, frames.shape[1], self.n_fft),
        )
        envelope = _overlap_add(envelope_frames, self.hop_length)[0]
        output = output / mx.maximum(envelope, 1e-11)[None]
        padding = (self.n_fft - self.hop_length) // 2
        return output[:, padding:-padding]


class ISTFTHead(nn.Module):
    def __init__(self, hidden_size: int, audio_patch_size: int):
        super().__init__()
        self.out = nn.Linear(hidden_size, audio_patch_size * 4 + 2)
        self.istft = ISTFT(audio_patch_size * 4, audio_patch_size)

    def __call__(self, value: mx.array) -> mx.array:
        prediction = self.out(value).astype(mx.float32)
        magnitude, phase = mx.split(prediction, 2, axis=-1)
        magnitude = mx.minimum(mx.exp(magnitude), 100.0)
        spectrum = magnitude * (mx.cos(phase) + 1j * mx.sin(phase))
        return self.istft(spectrum)


class RedAEAudioDecoder(nn.Module):
    def __init__(self, config: RedAEConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(
            config.bottleneck_dim,
            config.enc_extra_downsample_rate * config.dec_hidden_size,
        )
        self.qwen3 = Qwen3ASRTextModel(
            _qwen_config(
                hidden_size=config.dec_hidden_size,
                intermediate_size=config.dec_intermediate_size,
                num_hidden_layers=config.dec_num_hidden_layers,
                num_attention_heads=config.dec_num_attention_heads,
                num_key_value_heads=config.dec_num_key_value_heads,
                head_dim=config.head_dim,
                max_position_embeddings=config.dec_max_position_embeddings,
                vocab_size=config.vocab_size,
                rope_theta=config.rope_theta,
                sliding_window=config.dec_sliding_window,
                max_window_layers=config.dec_max_window_layers,
            )
        )
        self.istft_head = ISTFTHead(config.dec_hidden_size, config.audio_patch_size)

    def __call__(self, latents: mx.array) -> mx.array:
        if latents.ndim != 3 or int(latents.shape[-1]) != self.config.bottleneck_dim:
            raise ValueError(
                "RedAE decoder expects (batch, frames, bottleneck_dim), got "
                f"{latents.shape}"
            )
        value = self.in_proj(latents)
        value = value.reshape(
            value.shape[0],
            value.shape[1] * self.config.enc_extra_downsample_rate,
            self.config.dec_hidden_size,
        )
        value = self.qwen3(inputs_embeds=value).last_hidden_state
        return self.istft_head(value)


class RedAE(nn.Module):
    def __init__(self, config: RedAEConfig):
        super().__init__()
        self.config = config
        self.encoder = RedAEAudioEncoder(config)
        self.decoder = RedAEAudioDecoder(config)

    @classmethod
    def from_dir(cls, model_dir: str | Path) -> "RedAE":
        root = Path(model_dir)
        artifact = FireRedTTS3Config.from_dir(root)
        model = cls(RedAEConfig.from_artifact(artifact))
        weights = mx.load(root / artifact.files["redae"])
        model.load_weights(list(weights.items()), strict=True)
        return model

    @property
    def sample_rate(self) -> int:
        return self.config.audio_sample_rate

    @property
    def downsample_rate(self) -> int:
        return self.encoder.downsample_rate

    def pad_audio(self, audio: mx.array, *, multiple: int | None = None) -> mx.array:
        divisor = self.downsample_rate if multiple is None else int(multiple)
        target = math.ceil(int(audio.shape[-1]) / divisor) * divisor
        return mx.pad(audio, ((0, 0), (target - int(audio.shape[-1]), 0)))

    def encode(self, audio: mx.array) -> mx.array:
        if audio.ndim == 1:
            audio = audio[None]
        return self.encoder(self.pad_audio(audio))

    def decode(self, latents: mx.array) -> mx.array:
        return self.decoder(latents)


__all__ = ["ISTFT", "RedAE", "RedAEConfig"]
