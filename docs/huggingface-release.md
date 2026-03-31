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

Quantization variants live as subfolders inside the model repo rather than as
separate repos.

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

## Release Checklist

Before uploading:

1. Verify the converted artifact loads from the local quantized path.
2. Run one short end-to-end inference or decode pass on that path.
3. Confirm the folder contains only the files intended for publication.
4. Confirm the Hugging Face repo card is present and accurate.

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
