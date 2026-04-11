---
library_name: mlx
pipeline_tag: text-to-speech
base_model: fishaudio/s2-pro
base_model_relation: quantized
license: other
license_name: fish-audio-research-license
license_link: https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md
language:
- ja
- en
- zh
tags:
- mlx
- tts
- speech
- fishaudio
- fish-s2-pro
- quantized
- int8
- apple-silicon
- bundled-codec
---

# Fish Audio S2 Pro — MLX 8-bit

This repository contains a self-contained MLX-native int8 conversion of Fish Audio
S2 Pro for local text-to-speech on Apple Silicon.

> Note
> This repo is a community mirror of the canonical MLX conversion maintained by
> [AppAutomaton](https://github.com/appautomaton) at
> [`appautomaton/fishaudio-s2-pro-8bit-mlx`](https://huggingface.co/appautomaton/fishaudio-s2-pro-8bit-mlx).

## Model Details

- Developed by: AppAutomaton
- Shared by: `mlx-community`
- Original MLX repo: [`appautomaton/fishaudio-s2-pro-8bit-mlx`](https://huggingface.co/appautomaton/fishaudio-s2-pro-8bit-mlx)
- Upstream model: [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro)
- Task: text-to-speech
- Runtime: MLX on Apple Silicon
- Precision: int8 main model weights with bundled MLX codec assets

## Bundle Contents

This bundle is self-contained and includes:

- `config.json`
- `model.safetensors`
- tokenizer files
- `codec-mlx/config.json`
- `codec-mlx/model.safetensors`

The Fish S2 Pro runtime uses the bundled `codec-mlx/` directory to decode model
codes into waveform output.

## How to Get Started

Command-line generation with [`mlx-speech`](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate/fish_s2_pro.py \
  --text "Hello from Fish S2." \
  --model-dir /path/to/fishaudio-s2-pro-8bit-mlx \
  --output outputs/fish_s2_pro.wav
```

Fish S2 Pro supports inline prosody tags directly in the text input — e.g.
`[whisper]`, `[excited]`, `[laughing]`, `[pause]`. See the upstream repo for
the full tag set.

Minimal Python usage:

```python
from pathlib import Path

from mlx_speech.generation.fish_s2_pro import generate_fish_s2_pro

result = generate_fish_s2_pro(
    "Hello from Fish S2.",
    model_dir=Path("/path/to/fishaudio-s2-pro-8bit-mlx"),
)
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion keeps the Fish S2 Pro dual-autoregressive model architecture and
  ships a bundled MLX codec for waveform decode.
- The current bundle is intended for local MLX runtime use and parity validation.
- This mirror is a duplicated repo, not an automatically synchronized namespace mirror.

## Links

- Canonical MLX repo: [`appautomaton/fishaudio-s2-pro-8bit-mlx`](https://huggingface.co/appautomaton/fishaudio-s2-pro-8bit-mlx)
- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Fish Audio Research License — following the upstream license published with
[`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro).
