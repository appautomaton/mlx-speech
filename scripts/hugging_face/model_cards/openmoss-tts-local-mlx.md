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
- voice-cloning
- apple-silicon
- quantized
- 8bit
---

# OpenMOSS TTS — MLX

The smaller OpenMOSS text-to-speech model, converted and quantized for native MLX inference on Apple Silicon.

Supports direct synthesis, voice cloning from a short reference clip, and audio continuation. Fast and practical for single-speaker generation.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

Via [mlx-speech](https://github.com/appautomaton/mlx-speech):

**Generate speech:**
```bash
python scripts/generate_moss_local.py \
  --text "Hello, this is a test." \
  --output outputs/out.wav
```

**Clone a voice:**
```bash
python scripts/generate_moss_local.py \
  --mode clone \
  --text "This is a cloned voice." \
  --reference-audio reference.wav \
  --output outputs/clone.wav
```

```python
from mlx_speech.generation import MossTTSLocalModel

model = MossTTSLocalModel.from_path("mlx-int8")
```

## Model Details

Converted from the original OpenMOSS checkpoint with explicit MLX weight remapping and int8 quantization. Default runtime uses `W8Abf16` mixed precision with global + local KV cache enabled.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full runtime and conversion code.

## License

Apache 2.0 — following the upstream [OpenMOSS](https://github.com/open-moss) license terms.
