---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-speech
base_model: OpenMOSS-Team/MOSS-TTS-Local-Transformer
base_model_relation: quantized
tags:
- mlx
- tts
- speech
- voice-cloning
- apple-silicon
- quantized
- 8bit
---

# MOSS TTS Local Transformer — MLX 8-bit

This repository contains an MLX-native int8 conversion of MOSS TTS Local Transformer for single-speaker text-to-speech on Apple Silicon.

> Note
> This repo is a community mirror of the canonical MLX conversion maintained by
> [AppAutomaton](https://github.com/appautomaton) at
> [`appautomaton/openmoss-tts-local-mlx`](https://huggingface.co/appautomaton/openmoss-tts-local-mlx).

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## Model Details

- Developed by: AppAutomaton
- Shared by: `mlx-community`
- Original MLX repo: [`appautomaton/openmoss-tts-local-mlx`](https://huggingface.co/appautomaton/openmoss-tts-local-mlx)
- Upstream model: [`OpenMOSS-Team/MOSS-TTS-Local-Transformer`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer)
- Task: single-speaker text-to-speech and voice cloning
- Runtime: MLX on Apple Silicon

## How to Get Started

Command-line generation with [`mlx-speech`](https://github.com/appautomaton/mlx-speech):

**Generate speech:**
```bash
python scripts/generate/moss_local.py \
  --text "Hello, this is a test." \
  --output outputs/out.wav
```

**Clone a voice:**
```bash
python scripts/generate/moss_local.py \
  --mode clone \
  --text "This is a cloned voice." \
  --reference-audio reference.wav \
  --output outputs/clone.wav
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion keeps the original local TTS architecture and remaps weights explicitly for MLX inference.
- The default runtime path uses `W8Abf16` mixed precision with global and local KV cache enabled.
- This mirror is a duplicated repo, not an automatically synchronized namespace mirror.

## Links

- Canonical MLX repo: [`appautomaton/openmoss-tts-local-mlx`](https://huggingface.co/appautomaton/openmoss-tts-local-mlx)
- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-TTS-Local-Transformer`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer).
