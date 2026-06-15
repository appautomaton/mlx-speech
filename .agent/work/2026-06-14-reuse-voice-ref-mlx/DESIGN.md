# DESIGN: RE-USE voice-reference denoising (pure-MLX)

Scope: SPEC `.agent/work/2026-06-14-reuse-voice-ref-mlx/SPEC.md`. New model family
`reuse` (SEMamba), denoise-only, opt-in for DramaBox `denoise_ref=True`.

## Data flow (denoise-only)

```
noisy ref wav (mono)
  → [resample skipped: target_sr == in_sr for denoise-only]
  → chunk into Hann windows (chunk_size_s, 50% hop)
      per chunk:
        mag, pha = mag_phase_stft(chunk, n_fft, hop, win, compress_factor)
        amp_g, pha_g = SEMamba(mag, pha)
        amp_g = sweep_artifact_filter(amp_g)      # expm1(relu), zero-portion mask
        audio = mag_phase_istft(amp_g, pha_g, ...)
  → overlap-add normalize by window_sum
  → clean wav (clamped [-1, 1])
```

Reference: `.references/DramaBox/src/super_resolution.py` (the torch wrapper we
mirror). STFT params scale relative to the config training rate (8000):
`n_fft/hop/win = base * op_sr // base_sr`, made even.

## Components and file layout

```
src/mlx_speech/models/reuse/
  __init__.py
  stft.py            # mag_phase_stft / mag_phase_istft (compress_factor), sweep filter
  mamba/
    __init__.py
    scan.py          # selective scan (SSM recurrence), the core primitive
    block.py         # Mamba block (in/out proj, conv1d, gating) + bidirectional
  semamba.py         # SEMamba assembly (time/freq blocks, dense "d4" connections, heads)
  loader.py          # weight key mapping torch -> MLX

src/mlx_speech/generation/reuse.py   # REUSEEnhancer.from_dir(...).enhance(wav, in_sr)
scripts/convert/reuse.py             # nvidia/RE-USE ckpt -> MLX .safetensors
scripts/eval/reuse_capture_reference.py  # torch reference fixtures for parity (host-agnostic)
```

Hub: add `resolve_reuse_path(...)` + `REUSE_REPO = "appautomaton/<reuse-mlx>"` to
`_hub.py`, mirroring `resolve_gemma_backbone_path` / `resolve_codec_path`.

## Selective scan (the one real risk)

SEMamba uses `mamba_ssm`; upstream falls back to the pure-PyTorch
`selective_scan_ref` when CUDA kernels are absent. We port that reference
recurrence to MLX, not the CUDA kernel:

- Per channel/state: `h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t`,
  `y_t = sum_state(C_t * h_t) + D * x_t`, with `dt` from softplus.
- Sequential scan over time. Naive python-loop over T builds an O(T) graph;
  acceptable for short reference clips (~10s at the op rate, chunked at 1s).
  Associative/parallel scan is a later optimization, out of scope.
- Bidirectional: run forward, run on the time-flipped sequence, combine per
  SEMamba's bidirectional module. Confirm the exact combine from the reference
  in Slice 1.

## Parity strategy (two tiers, because mamba_ssm has no macOS wheels)

- **Tier A (Mac, unit tier):** MLX selective scan vs a self-written numpy
  reference recurrence; STFT round-trip; component output shapes. Runs anywhere,
  no external deps. This de-risks the hardest math without a torch/CUDA host.
- **Tier B (runtime tier, fixtures):** full MLX enhancer vs torch RE-USE output
  on a fixed noisy clip. `reuse_capture_reference.py` runs wherever torch +
  mamba_ssm are available (kernel-free path is fine) and writes small committed
  `.npz` fixtures under `tests/fixtures/reuse/`. The MLX test compares against
  the fixtures (no torch at test time). If fixtures cannot be generated on this
  Mac, capture is a human-action on another host.

Thresholds (set in Slice 6): waveform correlation >= 0.99 and bounded max-abs-diff.

## DramaBox integration

`generate(..., voice_ref, denoise_ref=True)`:
- Resolve + lazy-load the enhancer once; clean the reference waveform before
  `AudioProcessor.waveform_to_mel`; cache by `(path, sr)` (mirror
  `_denoise_voice_ref`). Everything downstream (mel, VAE, `apply_reference_latent`,
  per-token sigma) is unchanged.
- Default stays `False`. `denoise_ref=True` with the module/weights unavailable
  raises a clear error naming RE-USE and the opt-out (`denoise_ref=False`).
- No torch import on the runtime path.

## Licensing

`nvidia/RE-USE` weights are NSCLv1 (non-commercial). Converted MLX weights are
hosted on `appautomaton`; that repo carries the NSCLv1 license + NVIDIA
attribution + a non-commercial model card. Library code stays MIT. Precedent:
DramaBox (LTX-2 Community) and the Gemma backbone (Gemma terms) are already
hosted under their own restricted licenses.
