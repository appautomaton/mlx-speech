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

> Note
> This repo is a community mirror of the canonical MLX conversion maintained by
> [AppAutomaton](https://github.com/appautomaton) at
> [`appautomaton/longcat-audiodit-3.5b-8bit-mlx`](https://huggingface.co/appautomaton/longcat-audiodit-3.5b-8bit-mlx).

## Model Details

- Developed by: AppAutomaton
- Shared by: `mlx-community`
- Original MLX repo: [`appautomaton/longcat-audiodit-3.5b-8bit-mlx`](https://huggingface.co/appautomaton/longcat-audiodit-3.5b-8bit-mlx)
- Upstream model: [`meituan-longcat/LongCat-AudioDiT-3.5B`](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B)
- Task: text-to-speech
- Runtime: MLX on Apple Silicon
- Precision: int8 quantized weights with bundled tokenizer

## Bundle Contents

This bundle is self-contained and includes:

- `config.json`
- `model.safetensors`
- tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`)

## How to Get Started

Command-line generation with [`mlx-speech`](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate/longcat_audiodit.py \
  --text "Hello from LongCat AudioDiT." \
  --model-dir /path/to/longcat-audiodit-3.5b-8bit-mlx \
  --output-audio outputs/longcat.wav
```

Voice cloning:

```bash
python scripts/generate/longcat_audiodit.py \
  --text "Hello from LongCat AudioDiT." \
  --prompt-text "Original speaker text." \
  --prompt-audio /path/to/prompt.wav \
  --model-dir /path/to/longcat-audiodit-3.5b-8bit-mlx \
  --output-audio outputs/longcat_clone.wav \
  --guidance-method apg
```

Minimal Python usage:

```python
from pathlib import Path

from mlx_speech.generation.longcat_audiodit import generate_longcat_audiodit

result = generate_longcat_audiodit(
    text="Hello from LongCat AudioDiT.",
    model_dir=Path("/path/to/longcat-audiodit-3.5b-8bit-mlx"),
    output_audio="outputs/longcat.wav",
)
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion preserves the LongCat AudioDiT diffusion transformer and
  bundled VAE for waveform decode.
- The current bundle is intended for local MLX runtime use and parity validation.
- This mirror is a duplicated repo, not an automatically synchronized namespace mirror.

## Links

- Canonical MLX repo: [`appautomaton/longcat-audiodit-3.5b-8bit-mlx`](https://huggingface.co/appautomaton/longcat-audiodit-3.5b-8bit-mlx)
- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

MIT License — following the upstream license published with
[`meituan-longcat/LongCat-AudioDiT-3.5B`](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-3.5B).
