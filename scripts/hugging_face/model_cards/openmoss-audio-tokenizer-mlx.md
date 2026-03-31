---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: audio-to-audio
tags:
- mlx
- audio
- speech
- codec
- tokenizer
- apple-silicon
- quantized
- 8bit
---

# OpenMOSS Audio Tokenizer — MLX

The CAT codec component from the [OpenMOSS](https://github.com/open-moss) project, converted and quantized for native MLX inference on Apple Silicon.

This is a supporting model — it encodes and decodes audio tokens for the MOSS TTS model family. It is not a standalone TTS model.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

Load via [mlx-speech](https://github.com/appautomaton/mlx-speech):

```python
from mlx_speech.models.moss_audio_tokenizer import MossAudioTokenizerModel

model = MossAudioTokenizerModel.from_path("mlx-int8")
```

The tokenizer is loaded automatically when you run any MOSS TTS generation script. You typically do not need to load it directly.

```bash
python scripts/generate_moss_local.py \
  --text "Hello from mlx-speech." \
  --output outputs/out.wav
```

## Model Details

Converted from the original OpenMOSS checkpoint using explicit MLX weight remapping — no PyTorch at inference time. Quantized to int8 with `W8Abf16` mixed precision.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full conversion pipeline and runtime code.

## License

Apache 2.0 — following the upstream [OpenMOSS](https://github.com/open-moss) license terms.
