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
- voice-conditioned
- long-form
- diffusion
- apple-silicon
- quantized
- 8bit
---

# VibeVoice — MLX

VibeVoice converted and quantized for native MLX inference on Apple Silicon.

A hybrid LLM + diffusion architecture built for long-form speech and voice-conditioned generation. Works in greedy or sampled mode, and produces natural-sounding output at scale.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

Via [mlx-speech](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate_vibevoice.py \
  --text "Hello from VibeVoice." \
  --output outputs/vibevoice.wav
```

```python
from mlx_speech.generation import VibeVoiceModel

model = VibeVoiceModel.from_path("mlx-int8")
```

## Model Details

VibeVoice uses a 9B-parameter hybrid architecture combining a Qwen2 language model backbone with a continuous diffusion acoustic decoder. Converted to MLX with explicit weight remapping — no PyTorch at inference time.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full runtime and conversion code.

## License

Apache 2.0.
