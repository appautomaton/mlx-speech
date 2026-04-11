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

# OpenMOSS TTS Local Transformer — MLX 8-bit

MLX-native int8 conversion of OpenMOSS TTS Local Transformer for single-speaker TTS and voice cloning on Apple Silicon.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

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

## Model Details

- Upstream model: [`OpenMOSS-Team/MOSS-TTS-Local-Transformer`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer)
- Task: single-speaker text-to-speech and voice cloning
- Runtime: MLX on Apple Silicon, W8Abf16 mixed precision with KV cache

## Links

- Source code: [mlx-speech](https://github.com/appautomaton/mlx-speech)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-TTS-Local-Transformer`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer).
