---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-speech
base_model: OpenMOSS-Team/MOSS-TTSD-v1.0
base_model_relation: quantized
tags:
- mlx
- tts
- speech
- multi-speaker
- dialogue
- apple-silicon
- quantized
- 8bit
---

# OpenMOSS TTSD — MLX

MLX-native int8 conversion of OpenMOSS TTSD for multi-speaker dialogue generation on Apple Silicon.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

**Text must include `[S1]`/`[S2]` speaker tags. Omitting them produces degraded output.**

```bash
python scripts/generate/moss_ttsd.py \
  --text "[S1] Watson, I think we should go. [S2] Give me one moment." \
  --output outputs/dialogue.wav
```

Supported modes: `generation`, `continuation`, `voice_clone`, `voice_clone_and_continuation`.

```bash
python scripts/generate/moss_ttsd.py \
  --mode voice_clone \
  --text "[S1] This voice was cloned from the reference." \
  --prompt-audio-speaker1 reference.wav \
  --output outputs/clone.wav
```

Batch JSONL mode is also supported — see `python scripts/generate/moss_ttsd.py --help`.

## Model Details

- Upstream model: [`OpenMOSS-Team/MOSS-TTSD-v1.0`](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0)
- Task: multi-speaker text-to-speech and voice cloning
- Runtime: MLX on Apple Silicon

## Links

- Source code: [mlx-speech](https://github.com/appautomaton/mlx-speech)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-TTSD-v1.0`](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0).
