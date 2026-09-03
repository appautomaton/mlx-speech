# FireRedTTS3 Base MLX Plan

## Goal

Implement the approved [FireRedTTS3 Base MLX specification](SPEC.md) as a local BF16 voice-cloning pipeline that reaches 24 kHz waveform output.

## Architecture Approach

The runtime artifact is one flat Hugging Face model directory. It contains `config.json`, `core.safetensors`, `redae.safetensors`, `speaker.safetensors`, and the three tokenizer files at the same root. Components remain separate files to keep each file reasonably sized, but loading never requires a second model directory or repository.

Conversion writes MLX-native names and tensor layouts once. Linear weights keep their source orientation; Conv1d and other layout-sensitive weights are transposed during conversion rather than at runtime. Floating trainable weights are BF16. Integer state and any numerically required fixed statistics are preserved in their natural dtype and declared in `config.json`.

The FireRed implementation stays isolated under `src/mlx_speech/models/fireredtts3/` and reuses existing Qwen3/cache or audio utilities only where their equations and contracts match exactly. The official source remains the equation reference. The existing `.venv-torch` supplies one local MPS reference run; it is not imported by conversion or runtime code.

## Execution Routing and Topology

Execution is direct and serial: Slice 1 → Slice 2 → Slice 3 → Slice 4. Continue through every approved slice after its verification passes.

**Parallel-safe groups:** none.

No slice has a human checkpoint. The final local audio comparison is part of Slice 4 verification, not a separate implementation branch.

## Ordered Slice Sequence

### Slice 1: Flat BF16 artifact and strict conversion

**Objective:** Define the single-directory FireRedTTS3 artifact contract and reproducibly convert the pinned original weights into it.

**Acceptance criteria:**

- `models/firered/firered_tts3/mlx-bf16/` contains root-level `config.json`, `core.safetensors`, `redae.safetensors`, `speaker.safetensors`, `tokenizer.json`, `tokenizer_config.json`, and `vocab.json`; no component subdirectory is required.
- `config.json` identifies `fireredtts3_base`, BF16 storage policy, component files, architecture values, and pinned source revisions.
- Conversion validates the original 677 Base tensors, 458 RedAE tensors, and 937 CAM++ state entries before writing, applies explicit key/layout mapping, and rejects incomplete or incompatible inputs.
- Base and RedAE conversion uses MLX/safetensors only. The PyTorch `.bin` CAM++ archive is decoded without adding Torch to the project environment or published dependencies.
- Floating trainable weights are BF16; dtype exceptions are limited to declared non-trainable state required for correct inference.

**Touches:** `scripts/convert/fireredtts3.py`, `src/mlx_speech/models/fireredtts3/config.py`, `src/mlx_speech/models/fireredtts3/checkpoint.py`, `tests/unit/test_fireredtts3_config.py`, `tests/unit/test_fireredtts3_checkpoint.py`

**Produces:** the local flat `mlx-bf16` artifact and its strict loader metadata.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_fireredtts3_config.py tests/unit/test_fireredtts3_checkpoint.py
.venv/bin/python scripts/convert/fireredtts3.py --input-dir models/firered/firered_tts3/original --output-dir models/firered/firered_tts3/mlx-bf16
```

**Status:** complete
**Evidence:** added the flat artifact config, restricted torch-zip CAM++ reader,
MLX-native layout/dtype conversion, CLI, and 9 focused tests. The real converter
wrote 677-tensor `core.safetensors` (4,241,343,356 bytes), 458-tensor
`redae.safetensors` (1,887,612,163 bytes), and 937-entry
`speaker.safetensors` (14,794,136 bytes) beside the three tokenizer files and
root config; repeated conversion passed.
**Risks / next:** none.

### Slice 2: MLX audio codec and speaker conditioning

**Objective:** Implement the MLX-native waveform conditioning and reconstruction path used by Base.

**Acceptance criteria:**

- RedAE performs the pinned 24 kHz waveform → 25 Hz × 64 latent encode and latent → 24 kHz ISTFT decode path with the converted weights.
- CAM++ performs the official mono 16 kHz, 80-bin Kaldi-fbank frontend and returns a 512-dimensional speaker embedding.
- Qwen3 blocks used inside RedAE, CLS downsampling, sliding-window behavior, normalization, Conv/TDNN layers, statistics pooling, and ISTFT equations have focused numerical tests rather than shape-only assertions.
- Full converted RedAE and speaker checkpoints load with exact expected-key coverage and execute on a short local waveform without non-finite values.

**Depends on:** Slice 1

**Touches:** `src/mlx_speech/models/fireredtts3/redae.py`, `src/mlx_speech/models/fireredtts3/speaker.py`, shared exact-match utilities if needed, `tests/unit/test_fireredtts3_redae.py`, `tests/unit/test_fireredtts3_speaker.py`, `tests/checkpoint/test_fireredtts3_audio_checkpoint.py`

**Produces:** MLX RedAE and CAM++ modules loaded from the flat artifact.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_fireredtts3_redae.py tests/unit/test_fireredtts3_speaker.py
.venv/bin/python -m pytest tests/checkpoint/test_fireredtts3_audio_checkpoint.py
```

### Slice 3: MLX autoregressive core and generation loop

**Objective:** Implement Base prompt conditioning and patch-by-patch Qwen3/DiT latent generation.

**Acceptance criteria:**

- The tokenizer constructs the exact `<|language|><|sot|>{reference_text}{text}<|eot|>` sequence and rejects unsupported language tags.
- PatchEncoder maps four 64-dimensional latent frames into one Qwen embedding; the Base Qwen3 backbone performs cached prefill and one-token continuation with bounded KV state.
- The stop head, two-patch history, cosine time schedule, Euler flow updates, CFG, deterministic seed handling, and prompt-latent removal match the pinned source equations.
- Tiny deterministic models verify cache and generation behavior; the real converted core checkpoint loads with exact key coverage and produces finite latent patches.

**Depends on:** Slice 2

**Touches:** `src/mlx_speech/models/fireredtts3/core.py`, `src/mlx_speech/models/fireredtts3/dit.py`, `src/mlx_speech/models/fireredtts3/patch_encoder.py`, `src/mlx_speech/models/fireredtts3/tokenizer.py`, `src/mlx_speech/generation/fireredtts3.py`, `tests/unit/test_fireredtts3_core.py`, `tests/unit/test_fireredtts3_generation.py`, `tests/checkpoint/test_fireredtts3_core_checkpoint.py`

**Produces:** an MLX generator that accepts prepared cloning inputs and returns decoded, prompt-trimmed waveform.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/test_fireredtts3_core.py tests/unit/test_fireredtts3_generation.py
.venv/bin/python -m pytest tests/checkpoint/test_fireredtts3_core_checkpoint.py
```

### Slice 4: Public API, local MPS comparison, and waveform completion

**Objective:** Connect the generator to the existing TTS interface and prove one complete local voice-cloning request.

**Acceptance criteria:**

- `mlx_speech.tts.load(models/firered/firered_tts3/mlx-bf16)` returns a FireRed adapter accepting target text, `reference_audio`, `reference_text`, language, seed, CFG, flow steps, stop threshold, and patch budget.
- The CLI exposes the required cloning inputs without changing behavior for existing families.
- A small dev-only command runs the pinned official Base through the existing `.venv-torch` on MPS with SDPA and the exact same reference audio, ASR-derived transcript, target text, and seed used by MLX.
- The MLX public path writes a finite, non-silent mono 24 kHz WAV; local ASR recovers the requested text and the output is audibly conditioned by the reference.
- Invalid inputs fail clearly, runtime-purity coverage includes FireRed, documentation records usage and limitations, and all required tests pass.

**Depends on:** Slice 3

**Touches:** `src/mlx_speech/tts/__init__.py`, `src/mlx_speech/tts/_registry.py`, `src/mlx_speech/tts/_adapters/fireredtts3.py`, `src/mlx_speech/tts/generate.py`, `scripts/audit/fireredtts3_reference.py`, `tests/unit/test_fireredtts3_adapter.py`, `tests/unit/test_fireredtts3_dependency_guard.py`, `tests/integration/test_fireredtts3.py`, `docs/fireredtts3.md`, `README.md`, `docs/index.md`

**Produces:** the complete local FireRedTTS3 Base MLX feature through API and CLI.

**Verification:**

```bash
.venv/bin/python -m pytest tests/unit/
DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg/lib PYTHONPATH=.references/FireRedTTS3 .venv-torch/bin/python scripts/audit/fireredtts3_reference.py --model-dir models/firered/firered_tts3/original --reference-audio /tmp/fireredtts3-mlx-port/reference.wav --reference-text "For Timothy was a spoiled cat, and he allowed no one." --text "你好，很高兴认识你。" --language Chinese --output /tmp/fireredtts3-mlx-port/reference-mps.wav
RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/test_fireredtts3.py
```

## Aggregate Verification Commands

| Scope | Command |
| --- | --- |
| Fast regression | `.venv/bin/python -m pytest tests/unit/` |
| Real checkpoint loading | `.venv/bin/python -m pytest tests/checkpoint/test_fireredtts3_audio_checkpoint.py tests/checkpoint/test_fireredtts3_core_checkpoint.py` |
| End-to-end waveform | `RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/test_fireredtts3.py` |
