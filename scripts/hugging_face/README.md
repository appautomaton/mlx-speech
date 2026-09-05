# Hugging Face Upload Journal

This folder contains upload scripts for publishing converted MLX artifacts to
the Hugging Face Hub under the `appautomaton` org.

## Naming Convention

Each model gets its own HF repo. Quantization variants live as subfolders
inside the repo — not as separate repos.

| Local path | HF repo | Notes |
| --- | --- | --- |
| `models/openmoss/moss_tts_local/` | `appautomaton/openmoss-tts-local-mlx` | OpenMOSS TTS smaller model |
| `models/openmoss/moss_audio_tokenizer/` | `appautomaton/openmoss-audio-tokenizer-mlx` | CAT codec / audio tokenizer |
| `models/openmoss/moss_ttsd/` | `appautomaton/openmoss-ttsd-mlx` | OpenMOSS dialogue model |
| `models/openmoss/moss_sound_effect/` | `appautomaton/openmoss-sound-effect-mlx` | OpenMOSS sound effect model |
| `models/vibevoice/` | `appautomaton/vibevoice-mlx` | VibeVoice long-form speech |
| `models/cohere/cohere_transcribe/` | `appautomaton/cohere-asr-mlx` | Cohere ASR transcription |
| `models/firered/firered_tts3/mlx-bf16/` | `appautomaton/fireredtts3-mlx` | Complete Base BF16 bundle at `base/mlx-bf16/` |

## FireRedTTS3 family repository

`appautomaton/fireredtts3-mlx` contains complete inference bundles under each
model variant. The current release is `base/mlx-bf16/`. Future Instruct
artifacts belong under `instruct/` and can be added without moving Base.

FireRedTTS3 uses BF16 trainable weights. Do not upload the FP32 source files in
`models/firered/firered_tts3/original/` or create an INT8 release. The Base bundle
retains the declared FP32 ISTFT window and speaker running statistics.

The release staging directory contains exactly these files:

```text
hf-release/
  README.md
  LICENSE
  base/mlx-bf16/
    config.json
    core.safetensors
    redae.safetensors
    speaker.safetensors
    tokenizer.json
    tokenizer_config.json
    vocab.json
```

The root card comes from `model_cards/appautomaton/fireredtts3-mlx.md`. The
license is copied from the pinned official FireRedTTS3 source. Stage the seven
artifact files from `mlx-bf16/` only, verify their SHA-256 values against the
golden fixture manifest, and validate waveform generation before uploading.

```bash
hf upload appautomaton/fireredtts3-mlx \
  models/firered/firered_tts3/hf-release . \
  --repo-type model \
  --include README.md --include LICENSE --include 'base/mlx-bf16/**' \
  --commit-message "Publish the complete FireRedTTS3 Base BF16 bundle"
```

The upload updates only the named paths. It does not remove other model
variants from the repository. Runtime downloads use the same explicit
`base/mlx-bf16` selection through the shared loader.

## Quantization Variants

Within each repo, artifacts are organized by quantization:

```
appautomaton/<model-name>-mlx/
  mlx-int8/
    config.json
    model.safetensors
  mlx-4bit/
    config.json
    model.safetensors
```

Only upload variants that have been validated end-to-end locally first.

## Upload Tool

Uploads use the `hf` CLI from `huggingface_hub`.

Install the release tooling locally with:

```bash
uv sync --group release
```

Then run the wrapper scripts from the repo environment. The wrappers prefer the
`hf` executable that lives next to the active Python interpreter and only fall
back to a global `hf` on `PATH` if no local one is available.

Run them individually — do not batch upload without verifying the artifact
loads cleanly first.
