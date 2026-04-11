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
python scripts/generate/vibevoice.py \
  --text "Hello from VibeVoice." \
  --diffusion-steps 20 \
  --diffusion-steps-fast 8 \
  --diffusion-warmup-frames 10 \
  --output outputs/vibevoice.wav
```

```python
from mlx_speech.generation.vibevoice import (
    VibeVoiceGenerationConfig,
    synthesize_vibevoice,
)
from mlx_speech.models.vibevoice.checkpoint import load_vibevoice_model
from mlx_speech.models.vibevoice.tokenizer import VibeVoiceTokenizer

loaded = load_vibevoice_model("mlx-int8", strict=False)
tokenizer = VibeVoiceTokenizer.from_path("mlx-int8")
result = synthesize_vibevoice(
    loaded.model,
    tokenizer,
    "Hello from VibeVoice.",
    config=VibeVoiceGenerationConfig(max_new_tokens=256),
)
```

## Model Details

VibeVoice uses a 9B-parameter hybrid architecture combining a Qwen2 language model backbone with a continuous diffusion acoustic decoder. Converted to MLX with explicit weight remapping — no PyTorch at inference time.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full runtime and conversion code.

## License

Apache 2.0.
