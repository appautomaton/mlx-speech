---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-speech
base_model: OpenMOSS-Team/MOSS-TTSD-v1.0
base_model_relation: quantized
tags:
- mlx
- tts
- speech
- multi-speaker
- dialogue
- apple-silicon
- quantized
- 8bit
---

# OpenMOSS TTSD — MLX

This repository contains an MLX-native int8 conversion of OpenMOSS TTSD for multi-speaker dialogue generation on Apple Silicon.

It is intended for local multi-speaker speech generation with [`mlx-speech`](https://github.com/appautomaton/mlx-speech), without a PyTorch runtime at inference time.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## Model Details

- Developed by: AppAutomaton
- Shared by: AppAutomaton on Hugging Face
- Upstream model: [`OpenMOSS-Team/MOSS-TTSD-v1.0`](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0)
- Task: multi-speaker text-to-speech
- Runtime: MLX on Apple Silicon

## How to Get Started

Command-line generation with [`mlx-speech`](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate/moss_ttsd.py \
  --text "[S1] Watson, I think we should go. [S2] Give me one moment." \
  --output outputs/dialogue.wav
```

Minimal Python usage:

```python
from mlx_speech.generation import MossTTSDelayModel

model = MossTTSDelayModel.from_path("mlx-int8")
```

Speaker turns are tagged with `[S1]` and `[S2]` in the input text.

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion keeps the original TTSD architecture and remaps weights explicitly for MLX inference.
- The current runtime path is designed around speaker-tagged dialogue input and shared codec decoding.

## Links

- Source code: [mlx-speech](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-TTSD-v1.0`](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0).
