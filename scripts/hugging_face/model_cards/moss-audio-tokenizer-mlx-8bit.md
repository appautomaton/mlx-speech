---
language:
- zh
- en
license: apache-2.0
library_name: mlx
pipeline_tag: feature-extraction
base_model: OpenMOSS-Team/MOSS-Audio-Tokenizer
base_model_relation: quantized
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

# MOSS Audio Tokenizer — MLX 8-bit

This repository contains an MLX-native int8 conversion of the MOSS Audio Tokenizer for Apple Silicon.

> Note
> This repo is a community mirror of the canonical MLX conversion maintained by
> [AppAutomaton](https://github.com/appautomaton) at
> [`appautomaton/openmoss-audio-tokenizer-mlx`](https://huggingface.co/appautomaton/openmoss-audio-tokenizer-mlx).

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## Model Details

- Developed by: AppAutomaton
- Shared by: `mlx-community`
- Original MLX repo: [`appautomaton/openmoss-audio-tokenizer-mlx`](https://huggingface.co/appautomaton/openmoss-audio-tokenizer-mlx)
- Upstream model: [`OpenMOSS-Team/MOSS-Audio-Tokenizer`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer)
- Task: audio tokenization and codec decoding
- Runtime: MLX on Apple Silicon

## How to Get Started

Load it directly with [`mlx-speech`](https://github.com/appautomaton/mlx-speech):

```python
from mlx_speech.models.moss_audio_tokenizer import MossAudioTokenizerModel

model = MossAudioTokenizerModel.from_path("mlx-int8")
```

The tokenizer is loaded automatically when you run OpenMOSS generation scripts. You usually do not need to instantiate it directly.

```bash
python scripts/generate_moss_local.py \
  --text "Hello from mlx-speech." \
  --output outputs/out.wav
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion remaps the original MOSS audio tokenizer weights explicitly for MLX inference.
- The artifact is shared by the OpenMOSS local TTS, TTSD, and SoundEffect runtime paths in this repo family.
- This mirror is a duplicated repo, not an automatically synchronized namespace mirror.

## Links

- Canonical MLX repo: [`appautomaton/openmoss-audio-tokenizer-mlx`](https://huggingface.co/appautomaton/openmoss-audio-tokenizer-mlx)
- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Apache 2.0 — following the upstream license published with [`OpenMOSS-Team/MOSS-Audio-Tokenizer`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer).
