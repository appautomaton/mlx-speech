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
- multi-speaker
- long-form
- diffusion
- apple-silicon
- quantized
- 8bit
---

# VibeVoice — MLX

VibeVoice Large converted and quantized for native MLX inference on Apple Silicon. Hybrid LLM + diffusion architecture for long-form speech, multi-speaker dialogue, and voice cloning.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

**Single speaker:**
```bash
python scripts/generate/vibevoice.py \
  --text "Hello from VibeVoice." \
  --output outputs/vibevoice.wav
```

**Multi-speaker dialogue** — speaker labels are 0-based:
```bash
python scripts/generate/vibevoice.py \
  --text "Speaker 0: Have you tried VibeVoice?
Speaker 1: Not yet. Does it need PyTorch?
Speaker 0: No. Pure MLX, runs locally on Apple Silicon.
Speaker 1: That is impressive." \
  --output outputs/dialogue.wav
```

**Voice cloning** — one reference WAV per speaker:
```bash
python scripts/generate/vibevoice.py \
  --text "Speaker 0: This is cloned from the reference." \
  --reference-audio-speaker0 ref_speaker0.wav \
  --output outputs/clone.wav
```

Up to 4 speakers supported: `--reference-audio-speaker0` through `--reference-audio-speaker3`.

**Default generation settings** (matching upstream):
- Greedy decoding (deterministic)
- Seed: 42
- Diffusion steps: 20

Add `--no-greedy` to enable temperature + top-p sampling.

## Model Details

VibeVoice uses a 9B-parameter hybrid architecture combining a Qwen2 language model backbone with a continuous diffusion acoustic decoder. Converted to MLX with explicit weight remapping — no PyTorch at inference time.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full runtime and conversion code.

## License

Apache 2.0.
