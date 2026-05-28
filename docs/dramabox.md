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

### Known caveats (follow-ups)

- **STG (Spatio-Temporal Guidance)** is not yet wired through the DiT block
  code. Setting ``stg_scale != 0`` silently falls back to CFG-only. This is
  the only deviation from the warm-server defaults — implementing it
  requires threading a per-block ``skip_self_attn`` flag through
  `LTXBlock → LTXAttention` (the perturbation that replaces the QK softmax
  with a value passthrough).
- **Voice-reference conditioning (IC-LoRA)** is out of scope for the v5
  smoke. The `AudioProcessor` (waveform→mel front-end) ships as a stub
  that raises ``NotImplementedError`` until Stage 7+.
- **Per-token sigma**: the DiT forward uses broadcast-per-batch sigma. For
  the no-voice-ref path this is correct (the denoise mask is uniform); for
  IC-LoRA conditioning the upstream code uses per-token sigma which is a
  follow-up alongside voice ref.

## Settings

| Parameter | Default | Warm-server reference |
| --- | --- | --- |
| `cfg_scale` | 2.5 | 2.5 |
| `stg_scale` | 0.0 (CFG-only) | 1.5 |
| `rescale_scale` | "auto" → 0.3 (at cfg=2.5) | "auto" |
| `modality_scale` | 1.0 | 1.0 |
| `steps` | 30 | 30 |
| `seed` | 42 | 42 |

## Usage

### Python

```python
from mlx_speech.generation.dramabox import DramaBoxModel

model = DramaBoxModel.from_dir(
    "models/dramabox",
    gemma_dir="models/gemma_3_12b_it_4bit",
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
    --dramabox-dir models/dramabox \
    --gemma-dir models/gemma_3_12b_it_4bit \
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

DramaBox loads from three local directories:

| Directory | Contents | Size |
| --- | --- | --- |
| `models/dramabox/` | `dramabox-dit-v1.safetensors` (DiT, 1457 keys, ~6.6 GB bf16) | 6.6 GB |
|  | `dramabox-audio-components.safetensors` (VAE + vocoder + connector + aggregate, 1462 keys, ~1.9 GB bf16) | 1.9 GB |
|  | `config.json`, `assets/silence_latent_frame.pt` (training metadata; not used) | small |
| `models/gemma_3_12b_it_4bit/` | MLX 4-bit affine Gemma 3 12B IT (1300 keys, group_size=64, ~6.2 GB) | 6.2 GB |

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
