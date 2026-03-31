---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-speech
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

The large OpenMOSS multi-speaker dialogue model, converted and quantized for native MLX inference on Apple Silicon.

Built for natural two-speaker conversations. Pass speaker-tagged text and it handles turn-taking, prosody, and voice separation automatically. The strongest current path for multi-speaker generation in mlx-speech.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

Via [mlx-speech](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate_moss_ttsd.py \
  --text "[S1] Watson, I think we should go. [S2] Give me one moment." \
  --output outputs/dialogue.wav
```

```python
from mlx_speech.generation import MossTTSDelayModel

model = MossTTSDelayModel.from_path("mlx-int8")
```

Speaker turns are tagged with `[S1]` and `[S2]` in the input text.

## Model Details

Converted from the original OpenMOSS TTSD checkpoint with explicit MLX weight remapping and int8 quantization. Runs on the shared `MossTTSDelay` architecture.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full runtime and conversion code.

## License

Apache 2.0 — following the upstream [OpenMOSS](https://github.com/open-moss) license terms.
