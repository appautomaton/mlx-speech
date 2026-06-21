"""RE-USE / SEMamba speech enhancer wrapper (pure-MLX).

`REUSEEnhancer.from_dir(...).enhance(waveform, in_sr)` cleans a noisy mono
waveform with the converted MLX SEMamba model. Denoise-only: the operating rate
equals the input rate, so there is no resample and no bandwidth extension.

Mirrors the chunked inference in
`.references/DramaBox/src/super_resolution.py:REUSEUpsampler.__call__`
(lines ~213-266) for the ``target_sr == in_sr`` branch:

    wave (mono)
      → [resample skipped: op_sr == in_sr]
      → chunk at chunk_size_s * op_sr, 50% Hann hop
          per chunk:
            mag, pha = mag_phase_stft(chunk, n_fft, hop, win, compress_factor)
            amp_g, pha_g, _ = SEMamba(mag, pha)
            amp_g = sweep_artifact_filter(amp_g)
            audio = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, compress_factor)
      → Hann overlap-add, normalized by the window sum
      → clamp to [-1, 1]

The chunk loop is the Slice 3 `chunked_hann_ola` helper; the per-chunk denoise is
the closure passed to it. Mono-in / mono-out: the DramaBox integration (Slice 7)
collapses its reference to mono before calling and re-expands channels itself.

Weights are the converted MLX SEMamba under ``models/reuse/mlx/`` (produced by
`scripts/convert/reuse.py`; NSCLv1 non-commercial, gitignored).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from ..models.reuse import load_mlx_semamba
from ..models.reuse.semamba import SEMamba
from ..models.reuse.stft import (
    chunked_hann_ola,
    mag_phase_istft,
    mag_phase_stft,
    stft_params_for,
    sweep_artifact_filter,
)

# RE-USE config ``compress_factor`` (`.references/RE-USE/config.json`). The STFT
# front end compresses with ``log1p`` and expands with ``expm1(relu(.))``.
_COMPRESS_FACTOR = "relu_log1p"


class REUSEEnhancer:
    """Loaded RE-USE / SEMamba enhancer, ready to call ``.enhance(wave, in_sr)``."""

    def __init__(self, model: SEMamba) -> None:
        self.model = model

    # ----- construction ----------------------------------------------------

    @classmethod
    def from_dir(cls, mlx_weights_dir: str | Path) -> "REUSEEnhancer":
        """Load the enhancer from a directory of converted MLX weights.

        Args:
            mlx_weights_dir: directory holding ``model.safetensors`` written by
                `scripts/convert/reuse.py` (defaults to ``models/reuse/mlx``).
        """
        model = load_mlx_semamba(Path(mlx_weights_dir))
        return cls(model)

    # ----- enhancement -----------------------------------------------------

    def enhance(
        self,
        waveform: mx.array,
        in_sr: int,
        *,
        chunk_size_s: float = 1.0,
    ) -> mx.array:
        """Denoise a mono waveform, returning a same-length mono waveform.

        Denoise-only: the operating rate is ``in_sr`` (no resample, no bandwidth
        extension). The output has the same number of samples as the input and is
        clamped to ``[-1, 1]``.

        Args:
            waveform: ``(T,)`` or ``(1, T)`` mono float waveform.
            in_sr: sample rate of ``waveform`` (also the operating rate).
            chunk_size_s: chunk length in seconds for the overlap-add loop.

        Returns:
            The enhanced waveform with the same shape and length as the input.
        """
        squeeze = waveform.ndim == 1
        wave = waveform[None, :] if squeeze else waveform
        wave = wave.astype(mx.float32)

        op_sr = in_sr
        n_fft, hop, win = stft_params_for(op_sr)
        chunk_size = int(chunk_size_s * op_sr)

        def _denoise_chunk(chunk: mx.array) -> mx.array:
            mag, pha = mag_phase_stft(
                chunk, n_fft, hop, win, compress_factor=_COMPRESS_FACTOR
            )
            amp_g, pha_g, _ = self.model(mag, pha)
            amp_g = sweep_artifact_filter(amp_g)
            return mag_phase_istft(
                amp_g, pha_g, n_fft, hop, win, compress_factor=_COMPRESS_FACTOR
            )

        enhanced = chunked_hann_ola(_denoise_chunk, wave, chunk_size)
        enhanced = mx.clip(enhanced, -1.0, 1.0)

        return enhanced[0] if squeeze else enhanced


__all__ = ["REUSEEnhancer"]
