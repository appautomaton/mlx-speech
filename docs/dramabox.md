# DramaBox (v5) — MLX-Native Inference

Resemble's flow-matching diffusion TTS, ported to a pure-MLX inference
runtime. Inputs a text prompt; outputs a 48 kHz stereo waveform.

## Status

- **Architecture**: end-to-end pipeline runs (Gemma 3 12B IT → prompt
  encoder → 48-block DiT → AudioVAE → BigVGAN + BWE → 48 kHz wav).
- **Checkpoints load cleanly**: 1457 DiT keys + 102 AudioVAE keys + 1227
  vocoder keys + 131 prompt-pipeline keys all map to MLX modules.
- **Pure MLX runtime**: no torch, no `mlx_lm`, no `transformers`. The dev
  parity venv (`.venv-torch/`) is used only for capture scripts.

### Supported guidance

- **STG (Spatio-Temporal Guidance)** is wired through the DiT block code. A
  perturbed forward pass replaces the audio self-attention on
  ``stg_blocks`` (block 29 by default) with a value passthrough, and the
  guider adds ``stg_scale * (cond - ptb)``. The default ``stg_scale`` is
  ``1.5``, matching the warm-server reference. Set ``stg_scale=0`` for the
  faster CFG-only path (skips the third forward pass per step).
- **Voice-reference denoising (``denoise_ref=True``)** cleans the input
  reference with the RE-USE / SEMamba enhancer before VAE conditioning, giving
  the cloning model a clean speaker anchor. It is **opt-in** (default
  ``False``) and pure-MLX. The enhancer weights
  ([`appautomaton/reuse-semamba-mlx`](https://huggingface.co/appautomaton/reuse-semamba-mlx))
  derive from `nvidia/RE-USE` and are **NSCLv1 non-commercial**, resolved
  lazily on first use; `denoise_ref=True` raises a clear error if they cannot
  load. The MLX port matches the torch reference at 0.9997 waveform
  correlation. The raw voice-reference path (``denoise_ref=False``) needs none
  of this: the `AudioProcessor` waveform→mel front-end and the appended
  reference latent (per-token denoise mask) are fully implemented.

### Known caveats (follow-ups)

- **Per-token sigma**: the DiT forward uses broadcast-per-batch sigma on the
  no-voice-ref path (correct, since the denoise mask is uniform) and the
  per-token timestep on the raw-ref path. No outstanding work here.

## Settings

| Parameter | Default | Warm-server reference |
| --- | --- | --- |
| `cfg_scale` | 2.5 | 2.5 |
| `stg_scale` | 1.5 | 1.5 |
| `rescale_scale` | "auto" → 0.3 (at cfg=2.5) | "auto" |
| `modality_scale` | 1.0 | 1.0 |
| `steps` | 30 | 30 |
| `seed` | 42 | 42 |
| `denoise_ref` | False (opt-in; RE-USE non-commercial) | True |

## Usage

### Python

The simplest path is the unified loader. `tts.load("dramabox")` downloads both
the DramaBox weights and the Gemma 3 12B backbone, and returns 48 kHz stereo:

```python
import mlx_speech

model = mlx_speech.tts.load("dramabox")
result = model.generate("The weather today will be sunny.", duration_seconds=5.0)
# result.waveform : mx.array [2, T_samples], result.sample_rate : 48000
```

For direct control over both checkpoints (and the diffusion parameters), use the
lower-level model:

```python
from mlx_speech.generation.dramabox import DramaBoxModel

model = DramaBoxModel.from_dir(
    "models/dramabox/mlx-bf16",
    gemma_dir="models/gemma_3_12b_it_backbone/mlx-4bit",
)
result = model.generate(
    'A woman speaks clearly, "The weather today will be sunny."',
    duration_s=5.0,
    cfg_scale=2.5,
)
# result.waveform : mx.array [2, T_samples] float32 in [-1, 1]
# result.sample_rate : 48000
```

### CLI

```bash
.venv/bin/python scripts/generate_dramabox.py \
    --dramabox-dir models/dramabox/mlx-bf16 \
    --gemma-dir models/gemma_3_12b_it_backbone/mlx-4bit \
    --prompt 'A woman speaks clearly.' \
    --duration 5.0 \
    --out outputs/dramabox_smoke.wav
```

## Architecture

```
text prompt
    │
    ▼
LTXVGemmaTokenizer
  (plain text, left-pad, max_length=1024, no chat template)
    │
    ▼
Gemma 3 12B IT (MLX 4-bit) forward
  (output all 49 hidden states: embed + 48 layers)
    │
    ▼
EmbeddingsProcessor.process_hidden_states:
  FeatureExtractorV2:
    stack hidden states → [B, T, 3840, 49]
    per-token RMS norm → [B, T, 188160] (zero pad positions)
    rescale × sqrt(2048/3840)
    audio_aggregate_embed(...) → [B, T, 2048]
  convert_to_additive_mask → [B, 1, 1, T] (-finfo.max for pad)
  Embeddings1DConnector:
    replace padded slots with tiled learnable_registers
    new mask = all-zero (every slot valid)
    8 × (rms_norm + gated self-attn + rms_norm + FFN) blocks
    final rms_norm
  → a_ctx [B, 1024, 2048]

target shape = target_shape_from_duration(duration_s)
  → AudioPatchifier → state.latent [B, T, 128]
  → LTX2Scheduler.execute(steps=30, tokens=128) → sigmas[31]
  → GaussianNoiser convex-mix → noised state

X0Model(LTXModel):
  for sigma_i in sigmas[:-1]:
    velocity = LTXModel(latent, a_ctx, sigma_i)
    cond_x0 = latent - velocity * sigma_i
    if cfg > 1: uncond_x0 = X0Model(latent, a_ctx_neg, sigma_i)
    pred = cond + (cfg-1)*(cond - uncond) + ...
    if rescale: factor = rescale*(cond.std/pred.std) + (1-rescale); pred *= factor
    pred = post_process_latent(pred, denoise_mask, clean_latent)
    velocity = (latent - pred) / sigma_i
    latent = latent + velocity * (sigma_next - sigma_i)

unpatchify → silence_prior_fix (T > 513)
    │
    ▼
AudioVAE.decode (per-channel-stats un-normalize → decoder)
  → mel [B, 2, T_mel, 64]
    │
    ▼     (fp32 compute)
main BigVGAN-v2 → 16 kHz stereo wav
    │
    ▼     (BWE has its own STFT n_fft=512, hop=80)
BWE BigVGAN-v2 → 48 kHz residual + sinc-resampled skip
    │
    ▼
clamp [-1, 1] → output.wav (48 kHz, stereo)
```

## Weights

DramaBox loads from two local directories:

| Directory | Contents | Size |
| --- | --- | --- |
| `models/dramabox/mlx-bf16/` | `dramabox-dit-v1.safetensors` (DiT, 1457 keys, ~6.6 GB bf16) | 6.6 GB |
|  | `dramabox-audio-components.safetensors` (VAE + vocoder + connector + aggregate, 1462 keys, ~1.9 GB bf16) | 1.9 GB |
|  | `config.json`, `assets/silence_latent_frame.pt` (training metadata; not used) | small |
| `models/gemma_3_12b_it_backbone/mlx-4bit/` | MLX 4-bit affine Gemma 3 12B IT backbone (1300 keys, group_size=64, ~6.2 GB) | 6.2 GB |

Weights are NOT in git. Download separately.

## Memory budget (32 GB Mac target)

| Component | Storage | Compute |
| --- | --- | --- |
| Gemma 3 12B IT (MLX 4-bit) | 6.2 GB | 6.2 GB |
| `audio_aggregate_embed` (bf16) | 0.77 GB | 0.77 GB |
| `audio_embeddings_connector` (bf16) | 0.81 GB | 0.81 GB |
| DiT (bf16) | 6.58 GB | 6.58 GB |
| AudioVAE (bf16) | 0.5 GB | 0.5 GB |
| Vocoder + BWE (bf16 storage / fp32 compute) | 0.8 GB | 1.6 GB transient |
| Per-step buffers (2 guidance passes × ~150 MB) | — | 0.3 GB transient |
| **Total persistent** | **~15.7 GB** | — |
| **Peak during generate** | — | **~17.3 GB** |

Comfortable on 32 GB. Tight on 24 GB (would need to evict modules between
phases). v5 ships for 32 GB+ Macs.

## References

- DramaBox upstream: `.references/DramaBox/` pinned at
  `a70a5818e103c1c9fef22409c1e0c707ebf4f8a7` (see `docs/references.md`).
- Plan: `plans/v5-dramabox.md` (locked at rev4, marked Done after the v5
  smoke test landed).
