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
