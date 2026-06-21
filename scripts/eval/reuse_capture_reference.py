"""Capture torch RE-USE (SEMamba) reference fixtures for the MLX parity gate.

This is the CAPTURE side of Slice 6. It runs under the dev torch venv
(`.venv-torch/bin/python`, torch + einops present; `mamba_ssm` NOT installed)
and writes small committed `.npz` fixtures under `tests/fixtures/reuse/`. The
pure-MLX parity test (`tests/runtime/test_reuse_parity.py`) loads those fixtures
and never imports torch.

Two fixtures, both with fixed inputs and small sizes:

1. `model_level.npz` - a fixed random ``(noisy_mag, noisy_pha)`` of shape
   ``[1, F, T]`` and the torch SEMamba outputs ``(amp_g, pha_g, com_g)``. Isolates
   the core model numerics from the STFT / overlap-add front end. ``com_g`` is the
   amplitude-weighted complex output ``amp * [cos(pha), sin(pha)]`` of shape
   ``[1, F, T, 2]`` - the physically meaningful quantity the model feeds
   downstream, so the parity gate scores complex correlation / relative error on
   it rather than on raw phase (raw phase is undefined where magnitude ~ 0).

2. `end_to_end.npz` - a real noisy-speech clip (nvidia/RE-USE
   ``noisy_audio/mic_test2.wav``, resampled to 16 kHz and trimmed) and the torch
   denoise-only enhanced output. Mirrors the ``target_sr == in_sr`` branch of
   `.references/DramaBox/src/super_resolution.py:REUSEUpsampler.__call__`
   (STFT -> SEMamba -> sweep filter -> iSTFT -> Hann overlap-add). Real speech is
   used (not a synthetic tone) so the enhanced output keeps real structure: a
   synthetic harmonic input is suppressed ~96x by the denoiser to near-silence,
   which makes waveform correlation dominated by phase noise on silent samples
   rather than by the front end. With real speech the enhanced energy is retained
   and the correlation is meaningful.

mamba_ssm has no macOS wheels, so the reference SEMamba is built from the
vendored sources under `.references/mamba_ssm/` using the kernel-free recipe in
`.references/DramaBox/src/super_resolution.py:104-150`
(`_ensure_mamba_ssm_importable`): stub `selective_scan_cuda`, point
`mamba_ssm.Mamba` at the vendored `mamba_simple.Mamba`, and redirect
`selective_scan_fn`/`mamba_inner_fn` to the pure-PyTorch reference
implementations.

Usage (from the repo root):

    .venv-torch/bin/python scripts/eval/reuse_capture_reference.py

Writes:
    tests/fixtures/reuse/model_level.npz
    tests/fixtures/reuse/end_to_end.npz
"""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

# Repo layout. This file lives at scripts/eval/, so the root is two levels up.
_ROOT = Path(__file__).resolve().parents[2]
_REFS = _ROOT / ".references"
_REUSE_DIR = _REFS / "RE-USE"
_MAMBA_DIR = _REFS / "mamba_ssm"
_ORIGINAL_WEIGHTS = _ROOT / "models" / "reuse" / "original" / "model.safetensors"
_FIXTURE_DIR = _ROOT / "tests" / "fixtures" / "reuse"

# Model-level fixture sizes. F=161 is the RE-USE STFT freq count at the 16 kHz
# operating rate (n_fft=640 -> 321 bins; the model pads internally, but the
# real chunk path produces 321 bins). Keep the fixture small and self-consistent
# instead: any (F, T) is valid for the model, and a few hundred KB cap is met
# with F=161, T=64. The model is fully convolutional/recurrent in F and T, so
# the shape only needs to be internally consistent, not tied to a real STFT.
_MODEL_F = 161
_MODEL_T = 64

# End-to-end fixture: a real noisy-speech clip from nvidia/RE-USE, resampled to
# 16 kHz and trimmed. The clip ships at 44.1 kHz / ~5.1 s; we resample and take
# the first _E2E_DURATION_S so the committed .npz stays small (~128 KB/array).
_E2E_SR = 16_000
_E2E_DURATION_S = 1.5
_E2E_HF_REPO = "nvidia/RE-USE"
_E2E_HF_FILE = "noisy_audio/mic_test2.wav"


def _install_kernel_free_mamba_ssm() -> None:
    """Make ``from mamba_ssm import Mamba`` resolve to the vendored CPU source.

    Mirrors `super_resolution.py:_ensure_mamba_ssm_importable`. mamba_ssm is not
    installed (no macOS wheels), so we build a minimal package namespace in
    ``sys.modules`` backed by the two vendored files:

      .references/mamba_ssm/mamba_simple.py            -> Mamba
      .references/mamba_ssm/selective_scan_interface.py -> selective_scan_ref,
                                                           mamba_inner_ref

    The vendored ``mamba_simple`` already falls back to ``F.conv1d`` when
    ``causal_conv1d_fn is None``, and to the reference selective-scan when the
    CUDA kernel is absent, so no CUDA / Triton is required.
    """
    import importlib.util

    if "mamba_ssm" in sys.modules:
        return

    # 1. Stub the unconditional ``import selective_scan_cuda`` in the interface.
    if "selective_scan_cuda" not in sys.modules:
        stub = types.ModuleType("selective_scan_cuda")

        def _missing(*_a, **_kw):  # pragma: no cover - safety net only
            raise NotImplementedError(
                "selective_scan_cuda kernel missing; the reference path "
                "(selective_scan_ref) should have been used instead."
            )

        stub.fwd = _missing
        stub.bwd = _missing
        sys.modules["selective_scan_cuda"] = stub

    # 2. Build the package skeleton so relative imports inside the vendored
    #    modules resolve: mamba_ssm, mamba_ssm.ops, mamba_ssm.modules.
    pkg = types.ModuleType("mamba_ssm")
    pkg.__path__ = []  # marks it as a package
    sys.modules["mamba_ssm"] = pkg

    ops_pkg = types.ModuleType("mamba_ssm.ops")
    ops_pkg.__path__ = []
    sys.modules["mamba_ssm.ops"] = ops_pkg

    modules_pkg = types.ModuleType("mamba_ssm.modules")
    modules_pkg.__path__ = []
    sys.modules["mamba_ssm.modules"] = modules_pkg

    # 3. Load the vendored selective_scan_interface as
    #    mamba_ssm.ops.selective_scan_interface.
    ssi_spec = importlib.util.spec_from_file_location(
        "mamba_ssm.ops.selective_scan_interface",
        _MAMBA_DIR / "selective_scan_interface.py",
    )
    ssi = importlib.util.module_from_spec(ssi_spec)
    sys.modules["mamba_ssm.ops.selective_scan_interface"] = ssi
    ssi_spec.loader.exec_module(ssi)

    # Kernel-free redirect: the public fast-path names point at the references.
    ssi.selective_scan_fn = ssi.selective_scan_ref
    ssi.mamba_inner_fn = ssi.mamba_inner_ref

    # 4. Load the vendored mamba_simple as mamba_ssm.modules.mamba_simple. It
    #    imports ``from mamba_ssm.ops.selective_scan_interface import
    #    selective_scan_fn, mamba_inner_fn`` at module load, so the redirect
    #    above must already be in place.
    ms_spec = importlib.util.spec_from_file_location(
        "mamba_ssm.modules.mamba_simple",
        _MAMBA_DIR / "mamba_simple.py",
    )
    mamba_simple = importlib.util.module_from_spec(ms_spec)
    sys.modules["mamba_ssm.modules.mamba_simple"] = mamba_simple
    ms_spec.loader.exec_module(mamba_simple)

    # mamba_simple binds the names by reference at import; rebind to the
    # reference impls so Mamba.forward uses the CPU path.
    mamba_simple.selective_scan_fn = ssi.selective_scan_ref
    mamba_simple.mamba_inner_fn = ssi.mamba_inner_ref
    mamba_simple.causal_conv1d_fn = None  # forces the F.conv1d fallback

    # 5. Expose Mamba at the package root: ``from mamba_ssm import Mamba``.
    pkg.Mamba = mamba_simple.Mamba


def _load_torch_semamba() -> "torch.nn.Module":
    """Build the torch SEMamba and load the local original weights (cpu, eval)."""
    _install_kernel_free_mamba_ssm()

    if str(_REUSE_DIR) not in sys.path:
        sys.path.insert(0, str(_REUSE_DIR))

    import json

    from models.generator_SEMamba_time_d4 import SEMamba  # type: ignore
    from safetensors.torch import load_file  # type: ignore

    cfg = json.loads((_REUSE_DIR / "config.json").read_text())

    model = SEMamba(cfg)
    state = load_file(str(_ORIGINAL_WEIGHTS))
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    model.train(False)
    return model, cfg


def _make_even(v: float) -> int:
    v = int(round(v))
    return v if v % 2 == 0 else v + 1


def _stft_params(cfg: dict, op_sr: int) -> tuple[int, int, int]:
    """STFT params scaled from the config training rate, as in super_resolution."""
    base_n_fft = cfg["stft_cfg"]["n_fft"]
    base_hop = cfg["stft_cfg"]["hop_size"]
    base_win = cfg["stft_cfg"]["win_size"]
    base_sr = cfg["stft_cfg"]["sampling_rate"]
    n_fft = _make_even(base_n_fft * op_sr // base_sr)
    hop = _make_even(base_hop * op_sr // base_sr)
    win = _make_even(base_win * op_sr // base_sr)
    return n_fft, hop, win


def _capture_model_level(model: "torch.nn.Module") -> dict[str, np.ndarray]:
    """Fixed random (noisy_mag, noisy_pha) -> torch SEMamba (amp_g, pha_g, com_g).

    ``com_g`` is the amplitude-weighted complex output
    ``amp_g * [cos(pha_g), sin(pha_g)]`` of shape ``[1, F, T, 2]``: the quantity
    the model actually feeds downstream (iSTFT). The parity gate scores complex
    correlation / relative error on ``com_g`` rather than on raw phase, because
    raw phase is undefined where magnitude ~ 0 and so is not a meaningful metric.
    """
    rng = np.random.default_rng(1234)
    # The model compresses magnitude with log1p upstream, so feed a non-negative
    # magnitude in the realistic compressed range and a phase in (-pi, pi].
    noisy_mag = rng.uniform(0.0, 2.0, size=(1, _MODEL_F, _MODEL_T)).astype(np.float32)
    noisy_pha = rng.uniform(-np.pi, np.pi, size=(1, _MODEL_F, _MODEL_T)).astype(
        np.float32
    )

    with torch.inference_mode():
        amp_g, pha_g, com_g = model(
            torch.from_numpy(noisy_mag), torch.from_numpy(noisy_pha)
        )

    return {
        "noisy_mag": noisy_mag,
        "noisy_pha": noisy_pha,
        "amp_g": amp_g.detach().cpu().numpy().astype(np.float32),
        "pha_g": pha_g.detach().cpu().numpy().astype(np.float32),
        # Amplitude-weighted complex output [1, F, T, 2] = amp * [cos, sin].
        "com_g": com_g.detach().cpu().numpy().astype(np.float32),
    }


def _real_noisy_speech(sr: int, duration_s: float) -> np.ndarray:
    """Load nvidia/RE-USE ``noisy_audio/mic_test2.wav``, mono, 16 kHz, trimmed.

    A real noisy-speech clip so the denoise output retains real speech structure.
    A synthetic harmonic input is suppressed ~96x by the denoiser to near-silence,
    which makes the e2e waveform correlation dominated by phase noise on silent
    samples instead of the front end; real speech keeps energy and makes the
    metric meaningful.

    Downloaded into the HF cache (not committed); only the derived .npz is
    committed. ``HF_HUB_DISABLE_XET=1`` per repo HF-publishing convention.
    """
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
    from huggingface_hub import hf_hub_download  # type: ignore

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    path = hf_hub_download(_E2E_HF_REPO, _E2E_HF_FILE)

    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:  # collapse to mono
        wav = wav.mean(axis=1)
    if file_sr != sr:
        wav = librosa.resample(
            wav.astype(np.float64), orig_sr=file_sr, target_sr=sr
        ).astype(np.float32)

    n = int(sr * duration_s)
    wav = wav[:n]
    if wav.shape[0] < n:  # pad if the clip is shorter than the trim window
        wav = np.pad(wav, (0, n - wav.shape[0]))
    return np.ascontiguousarray(wav, dtype=np.float32)


def _torch_pad_or_trim_to_match(
    reference: "torch.Tensor", target: "torch.Tensor", pad_value: float = 1e-8
) -> "torch.Tensor":
    """Mirror utils.util.pad_or_trim_to_match for the e2e capture."""
    ref_len = reference.shape[-1]
    tgt_len = target.shape[-1]
    if tgt_len > ref_len:
        return target[..., :ref_len]
    if tgt_len < ref_len:
        pad = torch.full(
            (*target.shape[:-1], ref_len - tgt_len),
            pad_value,
            dtype=target.dtype,
            device=target.device,
        )
        return torch.cat([target, pad], dim=-1)
    return target


def _capture_end_to_end(
    model: "torch.nn.Module", cfg: dict
) -> dict[str, np.ndarray]:
    """Fixed noisy waveform -> torch denoise-only enhanced output.

    Ports the ``target_sr == in_sr`` denoise path of
    `super_resolution.py:__call__`: STFT -> SEMamba -> sweep filter -> iSTFT ->
    Hann overlap-add, using the vendored RE-USE STFT functions so the front end
    matches the model exactly.
    """
    from models.stfts import mag_phase_istft, mag_phase_stft  # type: ignore

    compress_factor = cfg["model_cfg"]["compress_factor"]
    op_sr = _E2E_SR
    n_fft, hop, win = _stft_params(cfg, op_sr)

    wave_np = _real_noisy_speech(op_sr, _E2E_DURATION_S)
    wav = torch.from_numpy(wave_np)[None, :]  # (1, T)

    chunk_size = int(1.0 * op_sr)  # chunk_size_s == 1.0
    hop_length = int(0.5 * chunk_size)
    window = torch.hann_window(chunk_size)

    n_ch, total = wav.shape
    enhanced = torch.zeros_like(wav)
    window_sum = torch.zeros_like(wav)
    n_chunks = (
        max(1, math.ceil((total - chunk_size) / hop_length) + 1)
        if total > chunk_size
        else 1
    )

    with torch.inference_mode():
        for c in range(n_ch):
            ch_in = wav[c : c + 1]
            for i in range(n_chunks):
                start = i * hop_length
                end = min(start + chunk_size, total)
                chunk = ch_in[:, start:end]
                if chunk.shape[-1] < 2:
                    continue
                noisy_mag, noisy_pha, _ = mag_phase_stft(
                    chunk,
                    n_fft=n_fft,
                    hop_size=hop,
                    win_size=win,
                    compress_factor=compress_factor,
                    center=True,
                    addeps=False,
                )
                amp_g, pha_g, _ = model(noisy_mag, noisy_pha)
                # Sweep-artifact filter (matches super_resolution.py:251-254).
                mag = torch.expm1(torch.relu(amp_g))
                zero_portion = (mag == 0).sum(dim=1) / mag.shape[1]
                amp_g[:, :, (zero_portion > 0.5)[0]] = 0

                audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, compress_factor)
                audio_g = _torch_pad_or_trim_to_match(chunk, audio_g, pad_value=1e-8)

                w_slice = window[: audio_g.shape[-1]]
                enhanced[c : c + 1, start : start + audio_g.shape[-1]] += (
                    audio_g * w_slice
                )
                window_sum[c : c + 1, start : start + audio_g.shape[-1]] += w_slice

        mask = window_sum > 1e-8
        enhanced[mask] = enhanced[mask] / window_sum[mask]
        enhanced = enhanced.clamp(-1.0, 1.0)

    return {
        "input": wave_np,  # (T,)
        "enhanced": enhanced[0].detach().cpu().numpy().astype(np.float32),  # (T,)
        "in_sr": np.int64(op_sr),
    }


def main() -> None:
    if not _ORIGINAL_WEIGHTS.is_file():
        raise SystemExit(
            f"original RE-USE weights not found at {_ORIGINAL_WEIGHTS}. "
            "Download them per Slice 1 before capturing fixtures."
        )

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading torch SEMamba (kernel-free mamba_ssm) ...")
    model, cfg = _load_torch_semamba()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded SEMamba: {n_params / 1e6:.2f}M params")

    print("Capturing model-level fixture ...")
    model_level = _capture_model_level(model)
    model_path = _FIXTURE_DIR / "model_level.npz"
    np.savez_compressed(model_path, **model_level)
    print(
        f"  wrote {model_path} "
        f"({model_path.stat().st_size / 1024:.1f} KB) "
        f"amp_g{model_level['amp_g'].shape} pha_g{model_level['pha_g'].shape} "
        f"com_g{model_level['com_g'].shape}"
    )

    print("Capturing end-to-end fixture ...")
    e2e = _capture_end_to_end(model, cfg)
    e2e_path = _FIXTURE_DIR / "end_to_end.npz"
    np.savez_compressed(e2e_path, **e2e)
    print(
        f"  wrote {e2e_path} "
        f"({e2e_path.stat().st_size / 1024:.1f} KB) "
        f"input{e2e['input'].shape} enhanced{e2e['enhanced'].shape}"
    )

    print("Done.")


if __name__ == "__main__":
    main()
