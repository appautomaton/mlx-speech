# Parity Capture (Dev-Only)

Scripts under `dev/parity/` capture golden tensors from the upstream PyTorch
reference at module boundaries so we can verify the MLX port stage by stage.

**Nothing here is imported by `src/mlx_speech/`.** The runtime stays pure MLX.
This venv is for capture only.

## Venv

We reuse the existing `.venv-torch/` at the repo root. It already has the full
parity stack:

| Package | Version |
| --- | --- |
| Python | 3.13.12 |
| torch | 2.11.0 |
| torchaudio | 2.11.0 |
| transformers | 5.0.0 |
| safetensors | 0.7.0 |
| tokenizers | 0.22.2 |
| einops | 0.8.2 |
| scipy | 1.17.1 |

`.venv*/` is gitignored. The plan calls this venv `.venv-dev/`; we reuse
`.venv-torch/` because it already has the right surface and avoids a second
~3 GB of duplicated torch wheels.

## Layout

```
dev/parity/
  README.md                   (this file)
  dramabox/                   (v5 captures)
    capture_gemma.py          text → all 49 hidden states
    capture_prompt.py         text → a_ctx (FeatureExtractorV2 + aggregate + connector)
    capture_audio_processor.py WAV → mel front-end output
    capture_vae.py            WAV ↔ latent (encoder + decoder)
    capture_bigvgan.py        mel → 16 kHz waveform
    capture_bwe.py            16 kHz waveform → 48 kHz waveform
    capture_dit_velocity.py   (noisy, sigma, a_ctx) → velocity
    capture_full_pipeline.py  end-to-end with fixed seed
```

Fixtures land under `tests/fixtures/dramabox/*.safetensors` — kept small
(≤ 100 MB total, gitignored if checkpoint-shaped, but small parity tensors are
fine to keep).

## Invocation

```bash
.venv-torch/bin/python dev/parity/dramabox/capture_gemma.py \
    --gemma-dir <local-or-hf-path-to-bf16-gemma-3-12b-it> \
    --out tests/fixtures/dramabox/gemma_hidden_states.safetensors
```

Each script accepts a `--gemma-dir` / `--dramabox-dir` and writes a
`.safetensors` bundle of `{name: tensor}` for each capture point. We import
upstream torch code by setting `PYTHONPATH=.references/DramaBox` rather than
vendoring it.
