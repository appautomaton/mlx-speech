# Upstream References

This repository uses `.references/` for optional local checkouts of upstream
projects that are useful for source inspection.

These checkouts are not part of the `mlx-speech` runtime, build, or packaging
story. They exist so implementation work can refer to upstream code locally
without turning those projects into vendored dependencies.

## Current Policy

- Keep upstream references shallow when possible.
- Prefer released code when a stable public release exists.
- Prefer current default-branch heads for fast-moving reference projects when we
  are studying implementation patterns rather than pinning a runtime dependency.

## Planned References

As of March 29, 2026:

- `mlx`: latest public GitHub release `v0.31.1`
- `MOSS-TTS`: shallow clone of `main`
- `MOSS-TTSD`: shallow clone of `main`
- `mlx-audio`: shallow clone of `main`
- `Step-Audio-EditX`: shallow clone of `main`

## Current Checkouts

- `.references/mlx`: `v0.31.1` at `ce45c52`
- `.references/MOSS-TTS`: `main` at `c74844ef6c08161160483c1bf3682235bdccae41`
- `.references/MOSS-TTSD`: `main` at `20dbb4fc44819435fee894d644a0402a0fee736a`
- `.references/mlx-audio`: `main` at `d28d68c6ac4e28f7d2d66007f640b06cf3fd8ceb` (v0.4.6, 2026-07-27)
  — Refreshed from `6408d2a` to pick up Nemotron 3.5 ASR support (PRs #771, #774
  cache-aware `stream_generate`, #775 shared `nemo/` package, #817 OOM fix).
  `mlx_audio/stt/models/nemotron_asr/` is the MLX-side reference for the v7 plan.
- `.references/dots.tts`: `main` at `5ed719e3d36f5a3f6d8037ca9a7009d4fd0520ba`
  (v0.2.1, 2026-07-06) — Official PyTorch source for dots.tts inference,
  fine-tuning, MeanFlow distillation, checkpoint behavior, and parity oracles.
  Read-only; never imported by the MLX runtime.
- `.references/dots-tts-mlx`: `main` at `f64479f51a2a9d7093533732cae86e765d8fb96e`
  (v0.7.0, 2026-06-10) — Pure-MLX, inference-only dots.tts reference supporting
  SOAR and MeanFlow checkpoints, voice cloning, waveform generation, conversion,
  and quantization. Read-only; not a runtime dependency.
- `.references/transformers`: `main` at `8213e0d920d52cb00dcade16b6d1f6e952ac0a8c` (sparse: `src/transformers/models/cohere_asr`, `src/transformers/models/moonshine`, `src/transformers/models/parakeet`)
- `.references/Step-Audio-EditX`: `main` at `8fa0a3e96979d3c47f6e6b531d234ff98acac878`
- `.references/DramaBox`: `main` at `a70a5818e103c1c9fef22409c1e0c707ebf4f8a7` (2026-05-23)
  — Resemble's flow-matching diffusion TTS. Source-truth for the v5 plan.
  Vendors a subset of LTX-2 (`ltx2/ltx_core`, `ltx2/ltx_pipelines`) as the
  diffusion framework. Read-only.
- `.references/granite-4.0-1b-speech`: `main` at `bd87ab862416353633ea431fe49b1614003623c5` (2026-04-02)
  — Hugging Face model repo for IBM Granite Speech 4.0 1B. Shallow clone with
  Git LFS smudge disabled; large model files remain as LFS pointers.
- `.references/Qwen3-ASR`: `main` at `c17a131fe028b2e428b6e80a33d30bb4fa57b8df` (2026-01-30)
  — Qwen3-ASR source repo. Shallow clone for studying the transformers/vLLM
  reference implementation, processor, prompt handling, streaming wrapper, and
  forced aligner. Code reference only; model weights are not stored here, and
  this checkout is never imported by the MLX runtime.
- `.references/RE-USE`: `nvidia/RE-USE` (HF) at `761905064ea1ea882e015e20a64e2e9d28458890` (2026-06-15)
  — NVIDIA RE-USE / SEMamba universal speech enhancement (9.61M params,
  1416 keys). Source-truth for the v6 RE-USE MLX port (DramaBox `denoise_ref`).
  Code subset only (no weights, no sample audio); weights live gitignored at
  `models/reuse/original/`. License NSCLv1 (non-commercial). Read-only.
- `.references/NeMo`: `NVIDIA/NeMo` `main` at `2639d4bef8d1450782263a8f616242acfb6fecb9`
  (2026-07-27) — Source-truth for the v7 Nemotron 3.5 ASR port. Blobless sparse
  clone (~8 MB) limited to the ASR inference pipeline:
  `nemo/collections/asr/{models,modules,parts/submodules,parts/preprocessing,parts/utils}`.
  Training code, configs, and weights are excluded. Key files:
  `modules/conformer_encoder.py` (cache-aware `chunked_limited` masking),
  `modules/rnnt.py`, `parts/submodules/rnnt_greedy_decoding.py`,
  `parts/preprocessing/features.py`. Read-only.
- `nvidia/nemotron-3.5-asr-streaming-0.6b`: Hugging Face revision
  `f3d333391852ba876df169dcc9ba902d25b6ab0b` (2026-07-06), staged under
  `models/nvidia/nemotron_3_5_asr_streaming_0_6b/original/` (gitignored).
  The upstream repo ships both a 2.4 GB `.nemo` archive and a 2.6 GB
  Transformers `model.safetensors`; conversion uses the `.nemo` source. Its
  checkpoint contains 657 fp32 tensors / 638,030,384 parameters. The NeMo
  config declares four attention contexts (`[56,3]`, `[56,0]`, `[56,6]`,
  `[56,13]`); NVIDIA's model card additionally documents `[56,1]` as a 160 ms
  runtime mode. Governing terms are OpenMDW-1.1, not the NVIDIA Open Model
  License. Redistribution must retain the OpenMDW-1.1 text and applicable
  copyright/origin notices; an official license copy is staged beside the
  checkpoint.
- `.references/mamba_ssm`: `state-spaces/mamba` tag `v2.2.2` at `8ffd905c91d207f5c0cc84fc2a2fb748655094f0`
  — Two files only: `ops/selective_scan_interface.py` (`selective_scan_ref`,
  `mamba_inner_ref`) and `modules/mamba_simple.py` (`class Mamba`). The exact
  reference math the MLX selective-scan port mirrors, since `mamba_ssm` has no
  macOS wheels. Read-only.

## Notes

- `mlx` is a real dependency of the project, but the checkout in
  `.references/mlx` is for local source inspection only.
- `MOSS-TTS`, `MOSS-TTSD`, and `mlx-audio` are reference codebases, not runtime
  dependencies.
- `MOSS-TTS` appears to be the active family repository and is the best primary
  OpenMOSS reference point going forward.
- Step-Audio assets staged locally for runtime bring-up, conversion, and source
  inspection:
  - `models/stepfun/step_audio_editx/original/`
  - `models/stepfun/step_audio_tokenizer/original/`
