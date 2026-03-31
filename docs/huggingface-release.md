# Hugging Face Release Workflow

This document records the intended publication workflow for `mlx-voice`
quantized artifacts.

## Scope

Publish converted MLX artifacts to Hugging Face.

Do not commit large weight files into this repository.

## Artifact Boundaries

Speech model:

- source: `models/openmoss/moss_tts_local/mlx-int8/`
- target repo: one Hugging Face model repo for the MLX `MossTTSLocal`
  runtime artifact

Audio tokenizer:

- source: `models/openmoss/moss_audio_tokenizer/mlx-int8/`
- target repo: one Hugging Face model repo for the MLX Cat codec artifact

## Files To Publish

Speech artifact:

- `config.json`
- `model.safetensors`
- tokenizer assets required by `MossTTSLocalProcessor.from_path(...)`

Codec artifact:

- `config.json`
- `model.safetensors`

## README Expectations

Each Hugging Face repo should include:

- source upstream reference
- note that the artifact is converted for MLX runtime use
- expected runtime precision policy: `W8Abf16`
- minimal local usage example with `mlx-voice`

## Publication Steps

1. Verify local artifacts load through the default runtime.
2. Verify one short end-to-end synthesis run on the quantized path.
3. Create or select the destination Hugging Face model repo.
4. Upload the artifact directory with the `hf` CLI.
5. Re-run one local-path load using the published file layout if needed.

## Notes

- Keep speech and codec artifacts separate.
- Keep this repository code-only; publish weights through Hugging Face.
- Prefer explicit version notes when artifact contents change.
