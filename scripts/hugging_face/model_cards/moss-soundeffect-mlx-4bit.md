---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-audio
base_model: OpenMOSS-Team/MOSS-SoundEffect
base_model_relation: quantized
tags:
- mlx
- audio
- sound-effects
- audio-generation
- apple-silicon
- quantized
- 4bit
---

# MOSS SoundEffect — MLX 4-bit

This repository contains an MLX-native 4-bit conversion of MOSS SoundEffect for local text-to-audio generation on Apple Silicon.

> Note
> This repo is a community mirror of the canonical MLX conversion maintained by
> [AppAutomaton](https://github.com/appautomaton) at
> [`appautomaton/openmoss-sound-effect-mlx`](https://huggingface.co/appautomaton/openmoss-sound-effect-mlx).

## Variants

| Path | Precision |
| --- | --- |
| `mlx-4bit/` | 4-bit quantized weights |

## Model Details

- Developed by: AppAutomaton
- Shared by: `mlx-community`
- Original MLX repo: [`appautomaton/openmoss-sound-effect-mlx`](https://huggingface.co/appautomaton/openmoss-sound-effect-mlx)
- Upstream model: [`OpenMOSS-Team/MOSS-SoundEffect`](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect)
- Task: text-to-audio and sound-effect generation
- Runtime: MLX on Apple Silicon

## How to Get Started

Command-line generation with [`mlx-speech`](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate/moss_sound_effect.py \
  --ambient-sound "rolling thunder with steady rainfall on a metal roof" \
  --duration-seconds 8 \
  --output outputs/thunder.wav
```

Minimal Python usage:

```python
from mlx_speech.generation import MossSoundEffectModel

model = MossSoundEffectModel.from_path("mlx-4bit")
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion keeps the original SoundEffect architecture and remaps weights explicitly for MLX inference.
- This is the 4-bit variant, and the published folder layout reflects that in `mlx-4bit/`.
- This mirror is a duplicated repo, not an automatically synchronized namespace mirror.

## Links

- Canonical MLX repo: [`appautomaton/openmoss-sound-effect-mlx`](https://huggingface.co/appautomaton/openmoss-sound-effect-mlx)
- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-SoundEffect`](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect).
