"""STFT front end for the RE-USE / SEMamba enhancer (pure-MLX port).

Mirrors `.references/RE-USE/models/stfts.py` (`mag_phase_stft` /
`mag_phase_istft`) and the "sweep artifact" filter applied to the model
amplitude output in `.references/DramaBox/src/super_resolution.py:251-254`.

The reference builds on `torch.stft` / `torch.istft` with ``center=True``,
``pad_mode='reflect'``, ``normalized=False`` and a periodic Hann window. MLX
has no STFT primitive, so we reproduce the same semantics directly on
`mx.fft.rfft` / `mx.fft.irfft`:

    forward  : reflect-pad n_fft//2, frame at hop, window, rfft → mag, phase
    inverse  : irfft each frame, window, overlap-add, divide by the
               overlap-added window-squared envelope (COLA), trim center pad

Magnitude compression follows the config's ``compress_factor`` string. The
RE-USE checkpoint uses ``"relu_log1p"`` (see `.references/RE-USE/config.json`),
so the forward compresses with ``log1p`` and the inverse expands with
``expm1(relu(.))``; both directions are reproduced so the round-trip holds.

STFT params scale from the config training rate (``sampling_rate=8000``,
``n_fft=320``, ``hop=40``, ``win=320``): at an operating rate ``op_sr`` each
param is ``even(base * op_sr // 8000)``. `stft_params_for` exposes that.
"""

from __future__ import annotations

import mlx.core as mx

_EPS = 1e-10

# Config training-rate STFT params (`.references/RE-USE/config.json:stft_cfg`).
_BASE_SR = 8000
_BASE_N_FFT = 320
_BASE_HOP = 40
_BASE_WIN = 320


def _make_even(v: float) -> int:
    """Round to the nearest even int. Mirrors ``REUSEUpsampler._make_even``."""
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def stft_params_for(op_sr: int) -> tuple[int, int, int]:
    """Return ``(n_fft, hop, win)`` scaled from the 8 kHz training config.

    Mirrors the scaling in
    `.references/DramaBox/src/super_resolution.py:224-226`. For denoise-only
    use ``op_sr == in_sr``.
    """
    n_fft = _make_even(_BASE_N_FFT * op_sr // _BASE_SR)
    hop = _make_even(_BASE_HOP * op_sr // _BASE_SR)
    win = _make_even(_BASE_WIN * op_sr // _BASE_SR)
    return n_fft, hop, win


def hann_window(win_size: int) -> mx.array:
    """Periodic Hann window, matching ``torch.hann_window(win_size)``."""
    n = mx.arange(win_size, dtype=mx.float32)
    return 0.5 - 0.5 * mx.cos(2.0 * mx.pi * n / float(win_size))


def _frame(signal: mx.array, frame_length: int, hop: int) -> mx.array:
    """Slice ``(B, T)`` into overlapping frames ``(B, n_frames, frame_length)``."""
    total = signal.shape[-1]
    if total < frame_length:
        raise ValueError(
            f"signal length {total} is shorter than frame_length {frame_length}"
        )
    n_frames = 1 + (total - frame_length) // hop
    starts = mx.arange(n_frames) * hop
    idx = starts[:, None] + mx.arange(frame_length)[None, :]
    return signal[:, idx]


def _compress_mag(mag: mx.array, compress_factor) -> mx.array:
    """Forward magnitude compression. Mirrors `stfts.py:54-57`."""
    if compress_factor in ("log1p", "relu_log1p", "signed_log1p"):
        return mx.log1p(mag)
    return mx.power(mag, compress_factor)


def _expand_mag(mag: mx.array, compress_factor) -> mx.array:
    """Inverse magnitude expansion. Mirrors `stfts.py:78-85`."""
    if compress_factor == "log1p":
        return mx.expm1(mag)
    if compress_factor == "signed_log1p":
        return mx.sign(mag) * mx.expm1(mx.abs(mag))
    if compress_factor == "relu_log1p":
        return mx.expm1(mx.maximum(mag, 0.0))
    return mx.power(mx.maximum(mag, 0.0), 1.0 / compress_factor)


def mag_phase_stft(
    wave: mx.array,
    n_fft: int,
    hop: int,
    win: int,
    compress_factor=1.0,
    *,
    center: bool = True,
    addeps: bool = False,
) -> tuple[mx.array, mx.array]:
    """Compute compressed magnitude and phase via STFT.

    Mirrors `.references/RE-USE/models/stfts.py:mag_phase_stft` with
    ``center=True``, ``pad_mode='reflect'``, ``normalized=False``.

    Args:
        wave: ``(B, T)`` or ``(T,)`` waveform.
        n_fft: FFT size.
        hop: Hop length.
        win: Window length (the RE-USE config keeps ``win == n_fft``).
        compress_factor: Magnitude compression. ``"relu_log1p"`` for RE-USE.
        center: Reflect-pad by ``n_fft // 2`` on each side before framing.
        addeps: Add ``1e-10`` inside the magnitude/phase as upstream does.

    Returns:
        ``(mag, pha)`` each shaped ``(B, n_freqs, n_frames)``.
    """
    squeeze = wave.ndim == 1
    if squeeze:
        wave = wave[None, :]
    wave = wave.astype(mx.float32)

    pad = n_fft // 2
    if center:
        wave = _reflect_pad(wave, pad)

    window = hann_window(win)
    if win < n_fft:
        left = (n_fft - win) // 2
        right = n_fft - win - left
        window = mx.concatenate(
            [mx.zeros((left,)), window, mx.zeros((right,))]
        )

    frames = _frame(wave, n_fft, hop)  # (B, n_frames, n_fft)
    windowed = frames * window[None, None, :]
    spec = mx.fft.rfft(windowed, n=n_fft, axis=-1)  # (B, n_frames, n_freqs)
    spec = mx.swapaxes(spec, 1, 2)  # (B, n_freqs, n_frames)

    real = spec.real
    imag = spec.imag
    if not addeps:
        mag = mx.sqrt(real * real + imag * imag)
        pha = mx.arctan2(imag, real)
    else:
        mag = mx.sqrt(real * real + imag * imag + _EPS)
        pha = mx.arctan2(imag + _EPS, real + _EPS)

    mag = _compress_mag(mag, compress_factor)
    if squeeze:
        mag = mag[0]
        pha = pha[0]
    return mag, pha


def mag_phase_istft(
    mag: mx.array,
    pha: mx.array,
    n_fft: int,
    hop: int,
    win: int,
    compress_factor=1.0,
    *,
    center: bool = True,
) -> mx.array:
    """Inverse STFT reconstructing the waveform from magnitude and phase.

    Mirrors `.references/RE-USE/models/stfts.py:mag_phase_istft`. Performs the
    same overlap-add with window-squared (COLA) normalization that
    ``torch.istft`` applies, then trims the ``n_fft // 2`` center padding.

    Returns:
        ``(B, T)`` (or ``(T,)`` if inputs were 2-D) reconstructed waveform.
    """
    squeeze = mag.ndim == 2
    if squeeze:
        mag = mag[None, ...]
        pha = pha[None, ...]
    mag = mag.astype(mx.float32)
    pha = pha.astype(mx.float32)

    mag = _expand_mag(mag, compress_factor)
    real = mag * mx.cos(pha)
    imag = mag * mx.sin(pha)
    spec = real + 1j * imag  # (B, n_freqs, n_frames)
    spec = mx.swapaxes(spec, 1, 2)  # (B, n_frames, n_freqs)

    frames = mx.fft.irfft(spec, n=n_fft, axis=-1).real  # (B, n_frames, n_fft)

    window = hann_window(win)
    if win < n_fft:
        left = (n_fft - win) // 2
        right = n_fft - win - left
        window = mx.concatenate(
            [mx.zeros((left,)), window, mx.zeros((right,))]
        )

    frames = frames * window[None, None, :]
    n_frames = frames.shape[1]
    out_len = n_fft + hop * (n_frames - 1)

    signal = _overlap_add(frames, hop, out_len)
    win_env = _overlap_add(
        mx.broadcast_to((window * window)[None, None, :], (1, n_frames, n_fft)),
        hop,
        out_len,
    )[0]
    signal = signal / mx.maximum(win_env, _EPS)[None, :]

    if center:
        pad = n_fft // 2
        signal = signal[:, pad : out_len - pad]
    if squeeze:
        signal = signal[0]
    return signal


def _reflect_pad(wave: mx.array, pad: int) -> mx.array:
    """Reflect-pad ``(B, T)`` by ``pad`` on each side (no edge repeat)."""
    if pad <= 0:
        return wave
    left = wave[:, 1 : pad + 1][:, ::-1]
    right = wave[:, -pad - 1 : -1][:, ::-1]
    return mx.concatenate([left, wave, right], axis=-1)


def _overlap_add(frames: mx.array, hop: int, out_len: int) -> mx.array:
    """Overlap-add ``(B, n_frames, frame_len)`` into ``(B, out_len)``."""
    b, n_frames, frame_len = frames.shape
    out = mx.zeros((b, out_len), dtype=frames.dtype)
    cols = mx.arange(frame_len)
    for i in range(n_frames):
        out[:, i * hop + cols] += frames[:, i]
    return out


def sweep_artifact_filter(amp: mx.array) -> mx.array:
    """Zero out spectral frames dominated by zeros ("sweep artifact" filter).

    Mirrors `.references/DramaBox/src/super_resolution.py:251-254`. The model
    amplitude ``amp`` is expanded with ``expm1(relu(.))``; for every time frame
    the fraction of zero-valued frequency bins is measured, and any frame whose
    zero fraction exceeds 0.5 has its amplitude column set to zero.

    Args:
        amp: ``(B, n_freqs, n_frames)`` model amplitude (compressed domain).

    Returns:
        ``amp`` with offending frames zeroed, same shape and dtype.
    """
    squeeze = amp.ndim == 2
    if squeeze:
        amp = amp[None, ...]

    mag = mx.expm1(mx.maximum(amp, 0.0))
    n_freq = mag.shape[1]
    zero_portion = (mag == 0).sum(axis=1) / n_freq  # (B, n_frames)
    keep = (zero_portion <= 0.5).astype(amp.dtype)  # (B, n_frames)
    out = amp * keep[:, None, :]

    if squeeze:
        out = out[0]
    return out


def chunked_hann_ola(
    process,
    wave: mx.array,
    chunk_size: int,
    *,
    hop_portion: float = 0.5,
    pad_value: float = 1e-8,
) -> mx.array:
    """Chunked Hann-window overlap-add over a long waveform.

    Mirrors the crossfade loop in
    `.references/DramaBox/src/super_resolution.py:228-265`: split ``wave`` into
    ``chunk_size`` windows hopped by ``hop_portion * chunk_size``, run
    ``process`` on each chunk, weight by a Hann analysis window, accumulate, and
    normalize by the summed window where it overlaps. The Slice 5 enhancer
    supplies ``process`` (denoise of one chunk).

    Args:
        process: Callable ``chunk (1, n) -> enhanced (1, m)``. Output is
            padded/trimmed back to the chunk length before windowing.
        wave: ``(1, T)`` or ``(T,)`` waveform on a single channel.
        chunk_size: Samples per chunk (``chunk_size_s * op_sr`` upstream).
        hop_portion: Fraction of ``chunk_size`` between chunk starts (0.5).
        pad_value: Fill value when an enhanced chunk is shorter than its input.

    Returns:
        ``(1, T)`` (or ``(T,)`` if input was 1-D) enhanced waveform.
    """
    squeeze = wave.ndim == 1
    if squeeze:
        wave = wave[None, :]
    wave = wave.astype(mx.float32)

    total = wave.shape[-1]
    hop_length = int(hop_portion * chunk_size)
    if hop_length < 1:
        raise ValueError("hop_portion * chunk_size must be >= 1")
    window = hann_window(chunk_size)

    if total > chunk_size:
        import math

        n_chunks = max(1, math.ceil((total - chunk_size) / hop_length) + 1)
    else:
        n_chunks = 1

    enhanced = mx.zeros_like(wave)
    window_sum = mx.zeros_like(wave)

    for i in range(n_chunks):
        start = i * hop_length
        end = min(start + chunk_size, total)
        chunk = wave[:, start:end]
        if chunk.shape[-1] < 2:  # skip degenerate tail
            continue

        out = process(chunk)
        out = _pad_or_trim_to_match(chunk, out, pad_value)

        seg = out.shape[-1]
        w_slice = window[:seg]
        cols = mx.arange(seg)
        enhanced[:, start + cols] += out * w_slice[None, :]
        window_sum[:, start + cols] += w_slice[None, :]

    mask = window_sum > 1e-8
    enhanced = mx.where(mask, enhanced / mx.maximum(window_sum, _EPS), enhanced)
    if squeeze:
        enhanced = enhanced[0]
    return enhanced


def _pad_or_trim_to_match(
    reference: mx.array, target: mx.array, pad_value: float = 1e-8
) -> mx.array:
    """Pad/trim ``target`` to ``reference`` length. Mirrors RE-USE util."""
    ref_len = reference.shape[-1]
    tgt_len = target.shape[-1]
    if tgt_len == ref_len:
        return target
    if tgt_len > ref_len:
        return target[:, :ref_len]
    pad = mx.full((target.shape[0], ref_len - tgt_len), pad_value, dtype=target.dtype)
    return mx.concatenate([target, pad], axis=-1)


__all__ = [
    "stft_params_for",
    "hann_window",
    "mag_phase_stft",
    "mag_phase_istft",
    "sweep_artifact_filter",
    "chunked_hann_ola",
]
