---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-audio
tags:
- mlx
- audio
- sound-effects
- audio-generation
- apple-silicon
- quantized
- 4bit
---

# OpenMOSS Sound Effect — MLX

The OpenMOSS sound effect model, converted and quantized for native MLX inference on Apple Silicon.

Generate ambient soundscapes, environmental audio, and sound effects directly from a text description. Runs entirely locally.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-4bit/` | 4-bit quantized weights |

## How to Get Started

Via [mlx-speech](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate_moss_sound_effect.py \
  --ambient-sound "rolling thunder with steady rainfall on a metal roof" \
  --duration-seconds 8 \
  --output outputs/thunder.wav
```

```python
from mlx_speech.generation import MossSoundEffectModel

model = MossSoundEffectModel.from_path("mlx-4bit")
```

## Model Details

Converted from the original OpenMOSS sound effect checkpoint with explicit MLX weight remapping and 4-bit quantization. Built on the shared `MossTTSDelay` architecture.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full runtime and conversion code.

## License

Apache 2.0 — following the upstream [OpenMOSS](https://github.com/open-moss) license terms.
