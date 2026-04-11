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

# OpenMOSS SoundEffect — MLX 4-bit

MLX-native 4-bit conversion of OpenMOSS SoundEffect for local text-to-audio generation on Apple Silicon.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-4bit/` | 4-bit quantized weights |

## How to Get Started

```bash
python scripts/generate/moss_sound_effect.py \
  --ambient-sound "rolling thunder with steady rainfall on a metal roof" \
  --duration-seconds 8 \
  --output outputs/thunder.wav
```

Duration controls the expected token budget at 12.5 tokens/second.

## Model Details

- Upstream model: [`OpenMOSS-Team/MOSS-SoundEffect`](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect)
- Task: text-to-audio and sound-effect generation
- Runtime: MLX on Apple Silicon

## Links

- Source code: [mlx-speech](https://github.com/appautomaton/mlx-speech)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-SoundEffect`](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect).
