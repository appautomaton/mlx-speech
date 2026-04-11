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

# MOSS TTSD v1.0 — MLX 8-bit

This repository contains an MLX-native int8 conversion of MOSS TTSD v1.0 for multi-speaker dialogue generation on Apple Silicon.

> Note
> This repo is a community mirror of the canonical MLX conversion maintained by
> [AppAutomaton](https://github.com/appautomaton) at
> [`appautomaton/openmoss-ttsd-mlx`](https://huggingface.co/appautomaton/openmoss-ttsd-mlx).

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## Model Details

- Developed by: AppAutomaton
- Shared by: `mlx-community`
- Original MLX repo: [`appautomaton/openmoss-ttsd-mlx`](https://huggingface.co/appautomaton/openmoss-ttsd-mlx)
- Upstream model: [`OpenMOSS-Team/MOSS-TTSD-v1.0`](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0)
- Task: multi-speaker text-to-speech
- Runtime: MLX on Apple Silicon

## How to Get Started

**Text must include `[S1]`/`[S2]` speaker tags. Omitting them produces degraded output.**

```bash
python scripts/generate/moss_ttsd.py \
  --text "[S1] Watson, I think we should go. [S2] Give me one moment." \
  --output outputs/dialogue.wav
```

Supported modes: `generation`, `continuation`, `voice_clone`, `voice_clone_and_continuation`. See `python scripts/generate/moss_ttsd.py --help`.

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion keeps the original TTSD architecture and remaps weights explicitly for MLX inference.
- The current runtime path is designed around speaker-tagged dialogue input and shared codec decoding.
- This mirror is a duplicated repo, not an automatically synchronized namespace mirror.

## Links

- Canonical MLX repo: [`appautomaton/openmoss-ttsd-mlx`](https://huggingface.co/appautomaton/openmoss-ttsd-mlx)
- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-TTSD-v1.0`](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0).
