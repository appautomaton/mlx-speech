"""Minimal MLX runtime for the Step-Audio vq02 tokenizer path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from .checkpoint import (
    StepAudioTokenizerAssets,
    load_step_audio_funasr_checkpoint,
    load_step_audio_tokenizer_assets,
)
from .config import StepAudioVQ02Config
from .processor import StepAudioTokenizerProcessor


@dataclass(frozen=True)
class StepAudioVQ02Checkpoint:
    model_dir: Path
    config: StepAudioVQ02Config
    state_dict: dict[str, mx.array]


@dataclass(frozen=True)
class StepAudioVQ02AlignmentReport:
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


def _load_cmvn(path: str | Path) -> np.ndarray:
    means_list: list[float] = []
    vars_list: list[float] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for index, raw_line in enumerate(lines):
        parts = raw_line.split()
        if not parts:
            continue
        if parts[0] == "<AddShift>" and index + 1 < len(lines):
            line_item = lines[index + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                means_list = [float(value) for value in line_item[3:-1]]
        if parts[0] == "<Rescale>" and index + 1 < len(lines):
            line_item = lines[index + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                vars_list = [float(value) for value in line_item[3:-1]]
    if not means_list or not vars_list:
        raise ValueError(f"Failed to parse CMVN file: {path}")
    return np.asarray([means_list, vars_list], dtype=np.float32)


def _next_power_of_two(value: int) -> int:
    return 1 if value == 0 else 2 ** (value - 1).bit_length()


def _get_strided(waveform: np.ndarray, window_size: int, window_shift: int, snip_edges: bool) -> np.ndarray:
    if snip_edges:
        if waveform.shape[0] < window_size:
            return np.empty((0, 0), dtype=waveform.dtype)
        frame_count = 1 + (waveform.shape[0] - window_size) // window_shift
    else:
        raise NotImplementedError("Only snip_edges=True is implemented for Step-Audio vq02.")
    stride = waveform.strides[0]
    return np.lib.stride_tricks.as_strided(
        waveform,
        shape=(frame_count, window_size),
        strides=(window_shift * stride, stride),
        writeable=False,
    ).copy()


def _feature_window_function(window_type: str, window_size: int) -> np.ndarray:
    if window_type == "hamming":
        return np.hamming(window_size).astype(np.float32)
    if window_type == "hanning":
        return np.hanning(window_size).astype(np.float32)
    if window_type == "povey":
        return np.hanning(window_size).astype(np.float32) ** 0.85
    if window_type == "rectangular":
        return np.ones((window_size,), dtype=np.float32)
    raise ValueError(f"Unsupported Step-Audio window type: {window_type}")


def _mel_scale(freq: np.ndarray) -> np.ndarray:
    return 1127.0 * np.log1p(freq / 700.0)


def _inverse_mel_scale(mel_freq: np.ndarray) -> np.ndarray:
    return 700.0 * (np.exp(mel_freq / 1127.0) - 1.0)


def _get_mel_banks(
    num_bins: int,
    window_length_padded: int,
    sample_freq: float,
    low_freq: float,
    high_freq: float,
) -> np.ndarray:
    num_fft_bins = window_length_padded // 2
    nyquist = sample_freq / 2.0
    high_freq = nyquist + high_freq if high_freq <= 0 else high_freq
    fft_bin_width = sample_freq / window_length_padded

    mel_low = float(_mel_scale(np.asarray(low_freq, dtype=np.float32)))
    mel_high = float(_mel_scale(np.asarray(high_freq, dtype=np.float32)))
    mel_delta = (mel_high - mel_low) / (num_bins + 1)

    bins = np.zeros((num_bins, num_fft_bins), dtype=np.float32)
    fft_mel = _mel_scale(fft_bin_width * np.arange(num_fft_bins, dtype=np.float32))
    for index in range(num_bins):
        left_mel = mel_low + index * mel_delta
        center_mel = mel_low + (index + 1.0) * mel_delta
        right_mel = mel_low + (index + 2.0) * mel_delta
        up = (fft_mel - left_mel) / (center_mel - left_mel)
        down = (right_mel - fft_mel) / (right_mel - center_mel)
        bins[index] = np.maximum(0.0, np.minimum(up, down))
    return bins


def _kaldi_fbank(
    waveform: np.ndarray,
    *,
    sample_frequency: int,
    frame_length_ms: float,
    frame_shift_ms: float,
    num_mel_bins: int,
    window_type: str,
    dither: float,
    remove_dc_offset: bool,
    preemphasis_coefficient: float,
    round_to_power_of_two: bool,
    snip_edges: bool,
    low_freq: float,
    high_freq: float,
    rng: np.random.Generator,
) -> np.ndarray:
    window_shift = int(sample_frequency * frame_shift_ms / 1000.0)
    window_size = int(sample_frequency * frame_length_ms / 1000.0)
    padded_window_size = _next_power_of_two(window_size) if round_to_power_of_two else window_size

    frames = _get_strided(waveform, window_size, window_shift, snip_edges)
    if frames.size == 0:
        return np.empty((0, num_mel_bins), dtype=np.float32)

    if dither != 0.0:
        frames = frames + rng.standard_normal(size=frames.shape, dtype=np.float32) * float(dither)

    if remove_dc_offset:
        frames = frames - frames.mean(axis=1, keepdims=True)

    if preemphasis_coefficient != 0.0:
        prev = np.pad(frames, ((0, 0), (1, 0)), mode="edge")[:, :-1]
        frames = frames - float(preemphasis_coefficient) * prev

    frames = frames * _feature_window_function(window_type, window_size)[None, :]
    if padded_window_size != window_size:
        frames = np.pad(frames, ((0, 0), (0, padded_window_size - window_size)), mode="constant")

    spectrum = np.abs(np.fft.rfft(frames, n=padded_window_size, axis=1)).astype(np.float32) ** 2
    mel_banks = _get_mel_banks(
        num_bins=num_mel_bins,
        window_length_padded=padded_window_size,
        sample_freq=float(sample_frequency),
        low_freq=float(low_freq),
        high_freq=float(high_freq),
    )
    mel_banks = np.pad(mel_banks, ((0, 0), (0, 1)), mode="constant")
    mel_energies = spectrum @ mel_banks.T
    return np.log(np.maximum(mel_energies, np.finfo(np.float32).eps)).astype(np.float32)


def _apply_cmvn(features: np.ndarray, cmvn: np.ndarray) -> np.ndarray:
    dim = features.shape[1]
    return ((features + cmvn[0:1, :dim]) * cmvn[1:2, :dim]).astype(np.float32)


def _apply_lfr(features: np.ndarray, lfr_m: int, lfr_n: int) -> np.ndarray:
    if features.shape[0] == 0:
        return features
    outputs: list[np.ndarray] = []
    total_frames = features.shape[0]
    left_padding = np.repeat(features[0:1], (lfr_m - 1) // 2, axis=0)
    padded = np.concatenate([left_padding, features], axis=0)
    padded_total = padded.shape[0]
    output_frames = int(np.ceil(total_frames / lfr_n))
    for index in range(output_frames):
        start = index * lfr_n
        end = start + lfr_m
        if end <= padded_total:
            chunk = padded[start:end]
        else:
            pad_count = end - padded_total
            chunk = np.concatenate(
                [padded[start:], np.repeat(padded[-1:], pad_count, axis=0)],
                axis=0,
            )
        outputs.append(chunk.reshape(1, -1))
    return np.concatenate(outputs, axis=0).astype(np.float32)


class StepAudioLayerNorm(nn.Module):
    def __init__(self, size: int, *, eps: float = 1e-12):
        super().__init__()
        self.weight = mx.ones((size,), dtype=mx.float32)
        self.bias = mx.zeros((size,), dtype=mx.float32)
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
        normalized = (x - mean) * mx.rsqrt(variance + self.eps)
        return normalized * self.weight + self.bias


class StepAudioDepthwiseConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.weight = mx.zeros((channels, kernel_size, 1), dtype=mx.float32)
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding

    def __call__(self, x: mx.array) -> mx.array:
        x = mx.pad(x, ((0, 0), (self.left_padding, self.right_padding), (0, 0)))
        return mx.conv1d(x, self.weight, stride=1, padding=0, groups=int(x.shape[-1]))


class StepAudioStreamSinusoidalPositionEncoder(nn.Module):
    def __call__(self, x: mx.array, cache: dict[str, Any] | None = None) -> mx.array:
        batch_size, timesteps, dim = map(int, x.shape)
        start_idx = int(cache.get("start_idx", 0)) if cache is not None else 0
        if cache is not None:
            cache["start_idx"] = start_idx + timesteps
        positions = np.arange(1, timesteps + start_idx + 1, dtype=np.float32)[None, :]
        log_increment = np.log(np.array([10000.0], dtype=np.float32)) / (dim / 2 - 1)
        inv_timescales = np.exp(np.arange(dim / 2, dtype=np.float32) * (-log_increment))
        inv_timescales = inv_timescales.reshape(batch_size, -1)
        scaled = positions.reshape(1, -1, 1) * inv_timescales.reshape(1, 1, -1)
        encoding = np.concatenate([np.sin(scaled), np.cos(scaled)], axis=2).astype(np.float32)
        return x + mx.array(encoding[:, start_idx : start_idx + timesteps], dtype=x.dtype)


class StepAudioPositionwiseFeedForward(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.w_1 = nn.Linear(input_dim, hidden_dim)
        self.w_2 = nn.Linear(hidden_dim, input_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w_2(mx.maximum(self.w_1(x), 0))


class StepAudioMultiHeadedAttentionSANM(nn.Module):
    def __init__(
        self,
        *,
        n_head: int,
        in_feat: int,
        n_feat: int,
        kernel_size: int,
    ):
        super().__init__()
        if n_feat % n_head != 0:
            raise ValueError(f"n_feat must be divisible by n_head, got {n_feat} and {n_head}.")
        self.h = n_head
        self.d_k = n_feat // n_head
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_q_k_v = nn.Linear(in_feat, n_feat * 3)
        self.fsmn_block = StepAudioDepthwiseConv1d(n_feat, kernel_size)

    def _forward_fsmn(self, inputs: mx.array) -> mx.array:
        x = self.fsmn_block(inputs)
        return x + inputs

    def _forward_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        batch, time, _ = map(int, x.shape)
        q_k_v = self.linear_q_k_v(x)
        split = int(self.h * self.d_k)
        q = q_k_v[..., :split]
        k = q_k_v[..., split : 2 * split]
        v = q_k_v[..., 2 * split :]
        q_h = q.reshape(batch, time, self.h, self.d_k).transpose(0, 2, 1, 3)
        k_h = k.reshape(batch, time, self.h, self.d_k).transpose(0, 2, 1, 3)
        v_h = v.reshape(batch, time, self.h, self.d_k).transpose(0, 2, 1, 3)
        return q_h, k_h, v_h, v

    def forward_chunk(
        self,
        x: mx.array,
        cache: dict[str, mx.array] | None = None,
        chunk_size: tuple[int, int, int] | list[int] | None = None,
        look_back: int = 0,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        q_h, k_h, v_h, v = self._forward_qkv(x)

        if chunk_size is not None and (look_back > 0 or look_back == -1):
            right_context = int(chunk_size[2])
            if cache is not None:
                k_h_stride = k_h[:, :, :-right_context, :] if right_context > 0 else k_h
                v_h_stride = v_h[:, :, :-right_context, :] if right_context > 0 else v_h
                k_h = mx.concatenate([cache["k"], k_h], axis=2)
                v_h = mx.concatenate([cache["v"], v_h], axis=2)
                next_cache_k = mx.concatenate([cache["k"], k_h_stride], axis=2)
                next_cache_v = mx.concatenate([cache["v"], v_h_stride], axis=2)
                if look_back != -1:
                    keep = look_back * int(chunk_size[1])
                    next_cache_k = next_cache_k[:, :, -keep:, :]
                    next_cache_v = next_cache_v[:, :, -keep:, :]
                cache = {"k": next_cache_k, "v": next_cache_v}
            else:
                prefix_k = k_h[:, :, :-right_context, :] if right_context > 0 else k_h
                prefix_v = v_h[:, :, :-right_context, :] if right_context > 0 else v_h
                cache = {"k": prefix_k, "v": prefix_v}

        fsmn_memory = self._forward_fsmn(v)
        q_h = q_h * (self.d_k ** -0.5)
        scores = q_h @ k_h.transpose(0, 1, 3, 2)
        attention = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        attended = attention @ v_h
        batch, _, time, _ = map(int, attended.shape)
        attended = attended.transpose(0, 2, 1, 3).reshape(batch, time, self.h * self.d_k)
        return self.linear_out(attended) + fsmn_memory, cache


class StepAudioEncoderLayerSANM(nn.Module):
    def __init__(
        self,
        *,
        in_size: int,
        size: int,
        self_attn: StepAudioMultiHeadedAttentionSANM,
        feed_forward: StepAudioPositionwiseFeedForward,
        normalize_before: bool = True,
    ):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = StepAudioLayerNorm(in_size)
        self.norm2 = StepAudioLayerNorm(size)
        self.in_size = in_size
        self.size = size
        self.normalize_before = normalize_before

    def forward_chunk(
        self,
        x: mx.array,
        cache: dict[str, mx.array] | None = None,
        chunk_size: tuple[int, int, int] | list[int] | None = None,
        look_back: int = 0,
    ) -> tuple[mx.array, dict[str, mx.array] | None]:
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        attn_out, cache = self.self_attn.forward_chunk(x, cache, chunk_size, look_back)
        if self.in_size == self.size:
            x = residual + attn_out
        else:
            x = attn_out
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)
        return x, cache


class StepAudioSANMEncoderChunkOpt(nn.Module):
    def __init__(self, config: StepAudioVQ02Config):
        super().__init__()
        self.config = config
        self._output_size = config.encoder.output_size
        self.embed = StepAudioStreamSinusoidalPositionEncoder()
        first_layer = StepAudioEncoderLayerSANM(
            in_size=config.encoder.input_size,
            size=config.encoder.output_size,
            self_attn=StepAudioMultiHeadedAttentionSANM(
                n_head=config.encoder.attention_heads,
                in_feat=config.encoder.input_size,
                n_feat=config.encoder.output_size,
                kernel_size=config.encoder.kernel_size,
            ),
            feed_forward=StepAudioPositionwiseFeedForward(
                config.encoder.output_size,
                config.encoder.linear_units,
            ),
            normalize_before=config.encoder.normalize_before,
        )
        self.encoders0 = [first_layer]
        self.encoders = [
            StepAudioEncoderLayerSANM(
                in_size=config.encoder.output_size,
                size=config.encoder.output_size,
                self_attn=StepAudioMultiHeadedAttentionSANM(
                    n_head=config.encoder.attention_heads,
                    in_feat=config.encoder.output_size,
                    n_feat=config.encoder.output_size,
                    kernel_size=config.encoder.kernel_size,
                ),
                feed_forward=StepAudioPositionwiseFeedForward(
                    config.encoder.output_size,
                    config.encoder.linear_units,
                ),
                normalize_before=config.encoder.normalize_before,
            )
            for _ in range(config.encoder.num_blocks - 1)
        ]
        self.after_norm = StepAudioLayerNorm(config.encoder.output_size)

    def output_size(self) -> int:
        return self._output_size

    def init_cache(
        self,
        *,
        chunk_size: tuple[int, int, int] | list[int],
        encoder_chunk_look_back: int,
    ) -> dict[str, Any]:
        feats_dim = self.config.encoder.input_size
        return {
            "start_idx": 0,
            "chunk_size": tuple(int(value) for value in chunk_size),
            "encoder_chunk_look_back": int(encoder_chunk_look_back),
            "last_chunk": False,
            "opt": None,
            "feats": mx.zeros((1, int(chunk_size[0]) + int(chunk_size[2]), feats_dim), dtype=mx.float32),
            "tail_chunk": False,
        }

    def _add_overlap_chunk(self, feats: mx.array, cache: dict[str, Any]) -> mx.array:
        overlap_feats = mx.concatenate([cache["feats"], feats], axis=1)
        keep = int(cache["chunk_size"][0]) + int(cache["chunk_size"][2])
        cache["feats"] = overlap_feats[:, -keep:, :]
        return overlap_feats

    def forward_chunk(
        self,
        xs_pad: mx.array,
        ilens: mx.array,
        *,
        cache: dict[str, Any],
    ) -> tuple[mx.array, mx.array]:
        xs_pad = xs_pad * math.sqrt(float(self.output_size()))
        xs_pad = self.embed(xs_pad, cache)
        if cache["tail_chunk"]:
            xs_pad = cache["feats"]
        else:
            xs_pad = self._add_overlap_chunk(xs_pad, cache)

        if cache["opt"] is None:
            new_cache: list[dict[str, mx.array] | None] = [None] * (len(self.encoders0) + len(self.encoders))
        else:
            new_cache = list(cache["opt"])

        for layer_idx, encoder_layer in enumerate(self.encoders0):
            xs_pad, new_cache[layer_idx] = encoder_layer.forward_chunk(
                xs_pad,
                new_cache[layer_idx],
                cache["chunk_size"],
                int(cache["encoder_chunk_look_back"]),
            )
        offset = len(self.encoders0)
        for layer_idx, encoder_layer in enumerate(self.encoders):
            xs_pad, new_cache[offset + layer_idx] = encoder_layer.forward_chunk(
                xs_pad,
                new_cache[offset + layer_idx],
                cache["chunk_size"],
                int(cache["encoder_chunk_look_back"]),
            )

        xs_pad = self.after_norm(xs_pad)
        if int(cache["encoder_chunk_look_back"]) > 0 or int(cache["encoder_chunk_look_back"]) == -1:
            cache["opt"] = new_cache
        return xs_pad, ilens


class StepAudioVQ02Model(nn.Module):
    def __init__(self, config: StepAudioVQ02Config):
        super().__init__()
        self.config = config
        self.encoder = StepAudioSANMEncoderChunkOpt(config)


class StepAudioVQ02Frontend:
    def __init__(self, config: StepAudioVQ02Config, cmvn: np.ndarray):
        self.config = config
        self.frontend = config.frontend
        self.cmvn = np.asarray(cmvn, dtype=np.float32)
        self.rng = np.random.default_rng(seed=0)

    def init_cache(self) -> dict[str, Any]:
        return {
            "reserve_waveforms": np.empty((1, 0), dtype=np.float32),
            "input_cache": np.empty((1, 0), dtype=np.float32),
            "lfr_splice_cache": [],
            "waveforms": None,
            "fbanks": None,
            "fbanks_lens": None,
        }

    def _compute_frame_num(self, sample_length: int) -> int:
        return int((sample_length - self.frontend.frame_length_samples) / self.frontend.frame_shift_samples + 1)

    def forward_fbank(
        self,
        input_values: np.ndarray,
        input_lengths: np.ndarray,
        *,
        cache: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = int(input_values.shape[0])
        assert batch_size == 1
        input_values = np.concatenate([cache["input_cache"], input_values], axis=1)
        frame_num = self._compute_frame_num(int(input_values.shape[-1]))
        tail = input_values.shape[-1] - frame_num * self.frontend.frame_shift_samples
        cache["input_cache"] = input_values[:, -tail:] if tail > 0 else np.empty((1, 0), dtype=np.float32)

        if frame_num <= 0:
            empty_feats = np.empty((0, 0, self.frontend.n_mels), dtype=np.float32)
            empty_lens = np.empty((0,), dtype=np.int32)
            cache["fbanks"] = empty_feats
            cache["fbanks_lens"] = empty_lens
            return np.empty((0, 0), dtype=np.float32), empty_feats, empty_lens

        waveforms: list[np.ndarray] = []
        feats: list[np.ndarray] = []
        feats_lens: list[int] = []
        used_length = (frame_num - 1) * self.frontend.frame_shift_samples + self.frontend.frame_length_samples
        for batch_idx in range(batch_size):
            waveform = input_values[batch_idx, :used_length]
            waveforms.append(waveform)
            features = _kaldi_fbank(
                waveform * (1 << 15),
                sample_frequency=self.frontend.sample_rate,
                frame_length_ms=self.frontend.frame_length_ms,
                frame_shift_ms=self.frontend.frame_shift_ms,
                num_mel_bins=self.frontend.n_mels,
                window_type=self.frontend.window_type,
                dither=self.frontend.dither,
                remove_dc_offset=self.frontend.remove_dc_offset,
                preemphasis_coefficient=self.frontend.preemphasis_coefficient,
                round_to_power_of_two=self.frontend.round_to_power_of_two,
                snip_edges=self.frontend.snip_edges,
                low_freq=self.frontend.low_freq,
                high_freq=self.frontend.high_freq,
                rng=self.rng,
            )
            feats.append(features)
            feats_lens.append(int(features.shape[0]))

        waveforms_batch = np.stack(waveforms).astype(np.float32)
        feats_pad = np.stack(feats).astype(np.float32)
        feats_lens_array = np.asarray(feats_lens, dtype=np.int32)
        cache["fbanks"] = feats_pad
        cache["fbanks_lens"] = feats_lens_array
        return waveforms_batch, feats_pad, feats_lens_array

    def forward_lfr_cmvn(
        self,
        features: np.ndarray,
        features_lengths: np.ndarray,
        *,
        is_final: bool,
        cache: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        outputs: list[np.ndarray] = []
        output_lengths: list[int] = []
        splice_indices: list[int] = []
        for batch_idx in range(int(features.shape[0])):
            current = features[batch_idx, : int(features_lengths[batch_idx]), :]
            total = int(current.shape[0])
            output_frames = int(np.ceil((total - (self.frontend.lfr_m - 1) // 2) / self.frontend.lfr_n))
            splice_idx = output_frames
            lfr_outputs: list[np.ndarray] = []
            for frame_idx in range(output_frames):
                start = frame_idx * self.frontend.lfr_n
                end = start + self.frontend.lfr_m
                if self.frontend.lfr_m <= total - start:
                    lfr_outputs.append(current[start:end].reshape(1, -1))
                else:
                    if is_final:
                        num_padding = self.frontend.lfr_m - (total - start)
                        frame = current[start:].reshape(-1)
                        for _ in range(num_padding):
                            frame = np.concatenate([frame, current[-1]], axis=0)
                        lfr_outputs.append(frame.reshape(1, -1))
                    else:
                        splice_idx = frame_idx
                        break
            if lfr_outputs:
                stacked = np.concatenate(lfr_outputs, axis=0).astype(np.float32)
            else:
                stacked = np.empty((0, self.frontend.n_mels * self.frontend.lfr_m), dtype=np.float32)
            stacked = _apply_cmvn(stacked, self.cmvn) if stacked.size else stacked
            outputs.append(stacked)
            output_lengths.append(int(stacked.shape[0]))
            splice_indices.append(min(total - 1, splice_idx * self.frontend.lfr_n))

            remainder = current[splice_indices[-1] :, :]
            cache["lfr_splice_cache"][batch_idx] = remainder

        max_length = max(output_lengths) if output_lengths else 0
        padded = np.zeros(
            (len(outputs), max_length, self.frontend.n_mels * self.frontend.lfr_m),
            dtype=np.float32,
        )
        for batch_idx, output in enumerate(outputs):
            if output.shape[0] > 0:
                padded[batch_idx, : output.shape[0], :] = output
        return padded, np.asarray(output_lengths, dtype=np.int32), np.asarray(splice_indices, dtype=np.int32)

    def __call__(
        self,
        input_values: np.ndarray,
        input_lengths: np.ndarray,
        *,
        is_final: bool,
        cache: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray]:
        waveforms, feats, feats_lengths = self.forward_fbank(input_values, input_lengths, cache=cache)
        batch_size = int(input_values.shape[0])

        if feats.shape[0] > 0 and feats.shape[1] > 0:
            reserve = cache["reserve_waveforms"]
            cache["waveforms"] = waveforms if reserve.size == 0 else np.concatenate([reserve, waveforms], axis=1)

            if not cache["lfr_splice_cache"]:
                repeat_count = (self.frontend.lfr_m - 1) // 2
                for batch_idx in range(batch_size):
                    cache["lfr_splice_cache"].append(np.repeat(feats[batch_idx, 0:1, :], repeat_count, axis=0))

            if int(feats_lengths[0]) + cache["lfr_splice_cache"][0].shape[0] >= self.frontend.lfr_m:
                lfr_cache = np.stack(cache["lfr_splice_cache"], axis=0)
                feats = np.concatenate([lfr_cache, feats], axis=1)
                feats_lengths = feats_lengths + lfr_cache.shape[1]
                frame_from_waveforms = int(
                    (cache["waveforms"].shape[1] - self.frontend.frame_length_samples)
                    / self.frontend.frame_shift_samples
                    + 1
                )
                minus_frame = (self.frontend.lfr_m - 1) // 2 if cache["reserve_waveforms"].size == 0 else 0
                feats, feats_lengths, lfr_indices = self.forward_lfr_cmvn(
                    feats,
                    feats_lengths,
                    is_final=is_final,
                    cache=cache,
                )
                reserve_frame_idx = int(lfr_indices[0]) - minus_frame
                cache["reserve_waveforms"] = cache["waveforms"][
                    :,
                    reserve_frame_idx * self.frontend.frame_shift_samples : frame_from_waveforms * self.frontend.frame_shift_samples,
                ]
                sample_length = (
                    (frame_from_waveforms - 1) * self.frontend.frame_shift_samples
                    + self.frontend.frame_length_samples
                )
                cache["waveforms"] = cache["waveforms"][:, :sample_length]
            else:
                cache["reserve_waveforms"] = cache["waveforms"][
                    :,
                    : -(self.frontend.frame_length_samples - self.frontend.frame_shift_samples),
                ]
                for batch_idx in range(batch_size):
                    cache["lfr_splice_cache"][batch_idx] = np.concatenate(
                        [cache["lfr_splice_cache"][batch_idx], feats[batch_idx, : int(feats_lengths[batch_idx]), :]],
                        axis=0,
                    )
                return np.empty((0, 0, self.config.encoder.input_size), dtype=np.float32), feats_lengths
        elif is_final:
            cache["waveforms"] = waveforms if cache["reserve_waveforms"].size == 0 else cache["reserve_waveforms"]
            feats = np.stack(cache["lfr_splice_cache"], axis=0)
            feats_lengths = np.zeros((batch_size,), dtype=np.int32) + feats.shape[1]
            feats, feats_lengths, _ = self.forward_lfr_cmvn(
                feats,
                feats_lengths,
                is_final=is_final,
                cache=cache,
            )
        return feats.astype(np.float32), feats_lengths.astype(np.int32)


def sanitize_step_audio_vq02_state_dict(weights: dict[str, mx.array]) -> dict[str, mx.array]:
    sanitized: dict[str, mx.array] = {}
    for key, value in weights.items():
        if not key.startswith("encoder."):
            continue
        if key.endswith(".fsmn_block.weight"):
            value = value.transpose(0, 2, 1)
        sanitized[key] = value.astype(mx.float32)
    return sanitized


def load_step_audio_vq02_checkpoint(model_dir: str | Path | None = None) -> StepAudioVQ02Checkpoint:
    assets = load_step_audio_tokenizer_assets(model_dir)
    config = StepAudioVQ02Config.from_config_yaml(assets.funasr_config_path)
    loaded = load_step_audio_funasr_checkpoint(model_dir)
    state_dict = sanitize_step_audio_vq02_state_dict(loaded.weights)
    return StepAudioVQ02Checkpoint(
        model_dir=assets.model_dir,
        config=config,
        state_dict=state_dict,
    )


def validate_step_audio_vq02_checkpoint_against_model(
    model: StepAudioVQ02Model,
    checkpoint: StepAudioVQ02Checkpoint,
) -> StepAudioVQ02AlignmentReport:
    model_params = tree_flatten(model.parameters(), destination={})
    model_keys = set(model_params)
    checkpoint_keys = set(checkpoint.state_dict)

    shape_mismatches: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for key in sorted(model_keys & checkpoint_keys):
        model_shape = tuple(int(dim) for dim in model_params[key].shape)
        checkpoint_shape = tuple(int(dim) for dim in checkpoint.state_dict[key].shape)
        if model_shape != checkpoint_shape:
            shape_mismatches.append((key, model_shape, checkpoint_shape))

    return StepAudioVQ02AlignmentReport(
        missing_in_model=tuple(sorted(checkpoint_keys - model_keys)),
        missing_in_checkpoint=tuple(sorted(model_keys - checkpoint_keys)),
        shape_mismatches=tuple(shape_mismatches),
    )


def load_step_audio_vq02_model(
    model_dir: str | Path | None = None,
    *,
    strict: bool = True,
) -> "LoadedStepAudioVQ02Model":
    assets = load_step_audio_tokenizer_assets(model_dir)
    checkpoint = load_step_audio_vq02_checkpoint(model_dir)
    model = StepAudioVQ02Model(checkpoint.config)
    report = validate_step_audio_vq02_checkpoint_against_model(model, checkpoint)
    if strict and not report.is_exact_match:
        raise ValueError(
            "Step-Audio vq02 checkpoint/model alignment failed: "
            f"{len(report.missing_in_model)} checkpoint-only keys, "
            f"{len(report.missing_in_checkpoint)} model-only keys, "
            f"{len(report.shape_mismatches)} shape mismatches."
        )
    model.load_weights(list(checkpoint.state_dict.items()), strict=strict)
    runtime = StepAudioVQ02Runtime(
        assets=assets,
        config=checkpoint.config,
        frontend=StepAudioVQ02Frontend(checkpoint.config, _load_cmvn(assets.funasr_model_dir / "am.mvn")),
        processor=StepAudioTokenizerProcessor(assets),
        model=model,
    )
    return LoadedStepAudioVQ02Model(
        assets=assets,
        config=checkpoint.config,
        checkpoint=checkpoint,
        model=model,
        alignment_report=report,
        runtime=runtime,
    )


@dataclass(frozen=True)
class LoadedStepAudioVQ02Model:
    assets: StepAudioTokenizerAssets
    config: StepAudioVQ02Config
    checkpoint: StepAudioVQ02Checkpoint
    model: StepAudioVQ02Model
    alignment_report: StepAudioVQ02AlignmentReport
    runtime: "StepAudioVQ02Runtime"


class StepAudioVQ02Runtime:
    def __init__(
        self,
        *,
        assets: StepAudioTokenizerAssets,
        config: StepAudioVQ02Config,
        frontend: StepAudioVQ02Frontend,
        processor: StepAudioTokenizerProcessor,
        model: StepAudioVQ02Model,
    ):
        self.assets = assets
        self.config = config
        self.frontend = frontend
        self.processor = processor
        self.model = model

    def init_cache(
        self,
        *,
        chunk_size: tuple[int, int, int] | list[int] | None = None,
        encoder_chunk_look_back: int | None = None,
    ) -> dict[str, Any]:
        chunk_size = tuple(chunk_size or self.assets.config.vq02_chunk_size)
        encoder_chunk_look_back = (
            self.assets.config.encoder_chunk_look_back
            if encoder_chunk_look_back is None
            else int(encoder_chunk_look_back)
        )
        return {
            "encoder": self.model.encoder.init_cache(
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
            ),
            "frontend": self.frontend.init_cache(),
            "prev_samples": np.empty((0,), dtype=np.float32),
        }

    def extract_encoder_features(
        self,
        waveform: np.ndarray,
        *,
        chunk_size: tuple[int, int, int] | list[int] | None = None,
        encoder_chunk_look_back: int | None = None,
        is_final: bool = True,
        cache: dict[str, Any] | None = None,
    ) -> np.ndarray:
        chunk_size = tuple(chunk_size or self.assets.config.vq02_chunk_size)
        encoder_chunk_look_back = (
            self.assets.config.encoder_chunk_look_back
            if encoder_chunk_look_back is None
            else int(encoder_chunk_look_back)
        )
        cache = self.init_cache(chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back) if cache is None else cache

        chunk_stride_samples = int(chunk_size[1] * 960)
        audio_sample = np.concatenate([cache["prev_samples"], waveform.astype(np.float32)])
        num_chunks = int(len(audio_sample) // chunk_stride_samples + int(is_final))
        tail = int(len(audio_sample) % chunk_stride_samples * (1 - int(is_final)))

        encoder_outs: list[np.ndarray] = []
        for chunk_index in range(num_chunks):
            chunk_final = bool(is_final and chunk_index == num_chunks - 1)
            audio_chunk = audio_sample[chunk_index * chunk_stride_samples : (chunk_index + 1) * chunk_stride_samples]
            if chunk_final and audio_chunk.shape[0] == 0:
                break
            if chunk_final and audio_chunk.shape[0] < 480:
                break

            speech, speech_lengths = self.frontend(
                audio_chunk[None, :],
                np.asarray([audio_chunk.shape[0]], dtype=np.int32),
                is_final=chunk_final,
                cache=cache["frontend"],
            )
            if speech.shape[0] == 0 and chunk_final:
                break
            if speech.shape[0] == 0:
                continue

            speech_mx = mx.array(speech, dtype=mx.float32)
            lengths_mx = mx.array(speech_lengths, dtype=mx.int32)
            encoder_out, _ = self.model.encoder.forward_chunk(
                speech_mx,
                lengths_mx,
                cache=cache["encoder"],
            )
            speech_len = int(speech_lengths[0])
            encoder_outs.append(np.asarray(encoder_out[:, -speech_len:, :], dtype=np.float32))

        cache["prev_samples"] = audio_sample[-tail:] if tail > 0 else np.empty((0,), dtype=np.float32)
        if not encoder_outs:
            return np.empty((0, self.config.encoder.output_size), dtype=np.float32)
        return np.concatenate(encoder_outs, axis=1)[0]

    def encode(
        self,
        audio: np.ndarray | mx.array,
        sample_rate: int,
        *,
        enable_trim: bool = True,
        energy_norm: bool = True,
    ) -> list[int]:
        waveform = self.processor.preprocess_wav(
            audio,
            sample_rate,
            enable_trim=enable_trim,
            energy_norm=energy_norm,
        )
        encoder_features = self.extract_encoder_features(waveform, is_final=True)
        return self.processor.cluster_linguistic_features(encoder_features)


__all__ = [
    "LoadedStepAudioVQ02Model",
    "StepAudioVQ02AlignmentReport",
    "StepAudioVQ02Checkpoint",
    "StepAudioVQ02Frontend",
    "StepAudioVQ02Model",
    "StepAudioVQ02Runtime",
    "load_step_audio_vq02_checkpoint",
    "load_step_audio_vq02_model",
    "sanitize_step_audio_vq02_state_dict",
    "validate_step_audio_vq02_checkpoint_against_model",
]
