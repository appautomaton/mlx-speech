---
library_name: mlx
pipeline_tag: text-to-speech
base_model: meituan-longcat/LongCat-AudioDiT-3.5B
base_model_relation: quantized
license: mit
language:
- zh
- en
tags:
- mlx
- tts
- speech
- longcat
- audiodit
- diffusion
- quantized
- int8
- apple-silicon
---

# LongCat AudioDiT 3.5B — MLX 8-bit

This repository contains a self-contained MLX-native int8 conversion of
LongCat AudioDiT 3.5B for local text-to-speech on Apple Silicon.

It is intended for local speech generation with
[`mlx-speech`](https://github.com/appautomaton/mlx-speech), without a PyTorch
runtime at inference time.

## Model Details

- Developed by: AppAutomaton
- Shared by: AppAutomaton on Hugging Face
- Upstream model: [`meituan-longcat/LongCat-AudioDiT-3.5B`](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B)
- Task: text-to-speech and voice cloning
- Runtime: MLX on Apple Silicon
- Precision: int8 quantized weights with bundled tokenizer

## Bundle Contents

This bundle is self-contained and includes:

- `config.json`
- `model.safetensors`
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)

## How to Get Started

**Basic generation:**

```bash
python scripts/generate/longcat_audiodit.py \
  --text "Hello from LongCat AudioDiT." \
  --output-audio outputs/longcat.wav
```

**Voice cloning:**

```bash
python scripts/generate/longcat_audiodit.py \
  --text "Hello from LongCat AudioDiT." \
  --prompt-text "Original speaker text." \
  --prompt-audio /path/to/prompt.wav \
  --output-audio outputs/longcat_clone.wav \
  --guidance-method apg
```

Minimal Python usage:

```python
from pathlib import Path

from mlx_speech.generation.longcat_audiodit import generate_longcat_audiodit

result = generate_longcat_audiodit(
    text="Hello from LongCat AudioDiT.",
    output_audio="outputs/longcat.wav",
)
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion preserves the LongCat AudioDiT diffusion transformer and
  bundled VAE for waveform decode.
- Voice cloning uses `--guidance-method apg` (Adaptive Projected Guidance) or
  `cfg` (Classifier-Free Guidance, default). `--guidance-strength` controls
  speaker adherence (default: 4.0).
- The current bundle is intended for local MLX runtime use and parity validation.

## Links

- Source code: [mlx-speech](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

MIT License — following the upstream license published with
[`meituan-longcat/LongCat-AudioDiT-3.5B`](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B).
