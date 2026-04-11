---
library_name: mlx
pipeline_tag: text-to-speech
base_model: fishaudio/s2-pro
base_model_relation: quantized
license: other
license_name: fish-audio-research-license
license_link: https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md
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

It is intended for local speech generation with
[`mlx-speech`](https://github.com/appautomaton/mlx-speech), without a PyTorch
runtime at inference time.

## Model Details

- Developed by: AppAutomaton
- Shared by: AppAutomaton on Hugging Face
- Upstream model: [`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro)
- Task: text-to-speech and voice cloning
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

**Basic generation:**

```bash
python scripts/generate/fish_s2_pro.py \
  --text "Hello from Fish S2 Pro." \
  --model-dir /path/to/fishaudio-s2-pro-8bit-mlx \
  --output outputs/fish_s2_pro.wav
```

**Voice cloning:**

```bash
python scripts/generate/fish_s2_pro.py \
  --text "This is a cloned voice." \
  --reference-audio /path/to/reference.wav \
  --reference-text "Transcript of the reference audio." \
  --model-dir /path/to/fishaudio-s2-pro-8bit-mlx \
  --output outputs/fish_s2_pro_clone.wav
```

**Inline prosody and emotion tags:**

Fish S2 Pro supports 15,000+ inline tags placed directly in the text. Tags are
single open-style `[tag]` — no closing tag. Place them immediately before the
word or phrase they apply to.

```bash
python scripts/generate/fish_s2_pro.py \
  --text "Now Bobby, [clearing throat] I need to talk to you. [whisper] This stays between us. [chuckle] Just kidding." \
  --reference-audio /path/to/reference.wav \
  --reference-text "Transcript of the reference audio." \
  --output outputs/fish_s2_pro_emotion.wav
```

Common tags: `[whisper]`, `[chuckle]`, `[laugh]`, `[clearing throat]`,
`[excited]`, `[sad]`, `[pause]`. See the upstream repo for the full tag list.

Minimal Python usage:

```python
from pathlib import Path

from mlx_speech.generation.fish_s2_pro import generate_fish_s2_pro

result = generate_fish_s2_pro(
    "Hello from Fish S2 Pro.",
    model_dir=Path("/path/to/fishaudio-s2-pro-8bit-mlx"),
)
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion keeps the Fish S2 Pro dual-autoregressive model architecture and
  ships a bundled MLX codec for waveform decode.
- Upstream defaults: temperature=0.8, top_p=0.8.
- The current bundle is intended for local MLX runtime use and parity validation.

## Links

- Source code: [mlx-speech](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Fish Audio Research License — following the upstream license published with
[`fishaudio/s2-pro`](https://huggingface.co/fishaudio/s2-pro).
