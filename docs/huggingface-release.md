# Hugging Face Release Workflow

This document describes how `mlx-speech` runtime artifacts are published to
Hugging Face.

The repository stays code-only. Converted weights, tokenizers, and processor
assets are published through Hugging Face model repos under the
`appautomaton` org.

## Published Repos

Each model family gets its own Hugging Face repo.

| Local model directory | Hugging Face repo | Published variant |
| --- | --- | --- |
| `models/openmoss/moss_audio_tokenizer/` | `appautomaton/openmoss-audio-tokenizer-mlx` | `mlx-int8` |
| `models/openmoss/moss_tts_local/` | `appautomaton/openmoss-tts-local-mlx` | `mlx-int8` |
| `models/openmoss/moss_ttsd/` | `appautomaton/openmoss-ttsd-mlx` | `mlx-int8` |
| `models/openmoss/moss_sound_effect/` | `appautomaton/openmoss-sound-effect-mlx` | `mlx-4bit` |
| `models/vibevoice/` | `appautomaton/vibevoice-mlx` | `mlx-int8` |
| `models/cohere/cohere_transcribe/` | `appautomaton/cohere-asr-mlx` | `mlx-int8` |
| `models/Qwen3-ASR-1.7B-MLX-BF16/` | `appautomaton/qwen3-asr-1.7b-bf16-mlx` | bf16, repo root |
| `models/dots_tts/` | `appautomaton/dots-tts-mlx` | SOAR/MF × `mlx-base`/`mlx-int8` |

Quantization variants live as subfolders inside the model repo rather than as
separate repos. Unquantized single-variant repos (Qwen3-ASR BF16) publish the
runtime package flat at the repo root instead.

## Release Boundaries

- Do not commit large checkpoint files into this repository.
- Keep each Hugging Face repo focused on one runtime artifact family.
- Keep original upstream checkpoints out of the published MLX repo layout.
- Only publish artifacts that have already been validated locally.

## Expected Repo Layout

Typical Hugging Face layout:

```text
appautomaton/<model-name>-mlx/
  README.md
  mlx-int8/
    config.json
    model.safetensors
    ...
```

Sound effect currently uses:

```text
appautomaton/openmoss-sound-effect-mlx/
  README.md
  mlx-4bit/
    config.json
    model.safetensors
    ...
```

Qwen3-ASR is unquantized BF16 with a single variant, so the package lives at
the repo root:

```text
appautomaton/qwen3-asr-1.7b-bf16-mlx/
  README.md
  config.json
  model.safetensors
  generation_config.json
  preprocessor_config.json
  chat_template.json
  tokenizer_config.json
  vocab.json
  merges.txt
```

dots.tts uses one family repository with four self-contained artifacts:

```text
appautomaton/dots-tts-mlx/
  README.md
  soar/
    mlx-base/
    mlx-int8/
  mf/
    mlx-base/
    mlx-int8/
```

The dots.tts upload target constructs only these four explicit include
patterns. It cannot select local `original/` checkpoints or obsolete
`mlx-bf16/` directories. Before publishing, validate hashes, required runtime
files, the model card, and the checked benchmark verdict without network access:

```bash
uv run python scripts/hugging_face/upload.py dots-tts --dry-run
```

The dry-run manifest must list exactly `soar/mlx-base`, `soar/mlx-int8`,
`mf/mlx-base`, and `mf/mlx-int8` under `appautomaton/dots-tts-mlx`. It fails if
an artifact digest no longer matches the passed quantization report.

When the local converted directory contains an upstream model card, exclude it
from the upload and publish this repo's own card instead.

The published subfolder should contain every runtime file required for loading
the artifact from a local clone of the Hugging Face repo.

## README Expectations

Each Hugging Face repo card should state:

- the upstream source model
- that the artifact is converted for MLX runtime use
- the published quantization variant
- the intended task or pipeline tag
- a minimal `mlx-speech` usage example
- any material license or usage restrictions
- a Links section pointing to the [source repo](https://github.com/appautomaton/mlx-speech)
  and the [project page](https://appautomaton.github.io/mlx-speech/)

## Release Checklist

Before uploading:

1. Verify the converted artifact loads from the local quantized path.
2. Run one short end-to-end inference or decode pass on that path.
3. Confirm the folder contains only the files intended for publication.
4. Confirm the Hugging Face repo card is present and accurate.
5. For dots.tts, run the required `--dry-run` gate and inspect all four remote
   paths before starting the resumable upload.

During upload:

1. Prefer the wrapper scripts in `scripts/hugging_face/`.
2. For large folders, prefer `hf upload-large-folder` so the upload can resume.
3. Do not run multiple uploads for the same folder in parallel.
4. Do not delete the local `.cache/huggingface/` upload state while an upload
   is active.

After upload:

1. Check the remote repo file list.
2. Confirm the quantized subfolder name is correct.
3. Confirm `model.safetensors` and all required config/tokenizer files landed.
4. Re-run one load test against the published file layout when practical.

## Operational Notes

- The upload wrappers are the canonical release entry points for this repo.
- When the standard `hf upload` path is unreliable for large files, use the
  resumable large-folder flow.
- If a model directory contains both published MLX artifacts and local
  reference material, stage only the publishable subfolder.
- Record meaningful artifact changes in the repo card or release notes when
  contents change.

## dots.tts Release Record — 2026-07-30

Published `appautomaton/dots-tts-mlx` at revision
`0af7ad2f837278b364902500d086553f1586ce9a`.

The authenticated remote inventory contained the Hub-generated
`.gitattributes`, the authoritative root `README.md`, and 56 runtime files under
exactly these prefixes:

- `soar/mlx-base/`
- `soar/mlx-int8/`
- `mf/mlx-base/`
- `mf/mlx-int8/`

No `original/`, `mlx-bf16/`, or sibling artifact path was present. The large
folder upload reported `56/56` files committed, 16.7 GB processed, and zero
ignored candidates.

Remote verification ran:

```bash
RUN_LOCAL_INTEGRATION=1 MLX_SPEECH_REQUIRE_CHECKPOINTS=1 \
  .venv/bin/python -m pytest -s tests/integration/test_dots_tts_hf.py
```

The runner additionally set a 16 GiB MLX memory limit. All four cases passed in
`988.95 s`: `dots-tts-soar-base`, `dots-tts-soar`, `dots-tts-mf-base`, and
`dots-tts-mf`. Each case used a new Hugging Face cache, materialized only its
selected artifact subtree and root README, strict-loaded the checkpoint, and
produced finite, non-silent mono 48 kHz continuation-clone waveform output.
Peak MLX allocation was `6,521,655,508` bytes; macOS process peak RSS was
`5,784,944,640` bytes. Generated/reference audio and downloaded caches remain
outside Git.
