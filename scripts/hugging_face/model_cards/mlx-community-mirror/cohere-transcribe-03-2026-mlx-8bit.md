---
language:
- en
license: apache-2.0
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: CohereLabs/cohere-transcribe-03-2026
base_model_relation: quantized
tags:
- mlx
- asr
- speech-recognition
- transcription
- apple-silicon
- quantized
- 8bit
---

# Cohere Transcribe 03-2026 — MLX 8-bit

This repository contains an MLX-native int8 conversion of Cohere Transcribe 03-2026 for local automatic speech recognition on Apple Silicon.

> Note
> This repo is a community mirror of the canonical MLX conversion maintained by
> [AppAutomaton](https://github.com/appautomaton) at
> [`appautomaton/cohere-asr-mlx`](https://huggingface.co/appautomaton/cohere-asr-mlx).

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## Model Details

- Developed by: AppAutomaton
- Shared by: `mlx-community`
- Original MLX repo: [`appautomaton/cohere-asr-mlx`](https://huggingface.co/appautomaton/cohere-asr-mlx)
- Upstream model: `CohereLabs/cohere-transcribe-03-2026`
- Task: automatic speech recognition
- Runtime: MLX on Apple Silicon

## How to Get Started

Command-line transcription with [`mlx-speech`](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/generate/cohere_asr.py \
  --audio input.wav \
  --output transcript.txt
```

Minimal Python usage:

```python
import numpy as np
import soundfile as sf

from mlx_speech.generation import CohereAsrModel

audio, sr = sf.read("input.wav", dtype="float32", always_2d=False)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if sr != 16000:
    old_len = len(audio)
    new_len = int(round(old_len * 16000 / sr))
    audio = np.interp(
        np.linspace(0, old_len - 1, new_len),
        np.arange(old_len),
        audio,
    ).astype(np.float32)

model = CohereAsrModel.from_path("mlx-int8")
result = model.transcribe(audio, sample_rate=16000, language="en")
print(result.text)
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The conversion keeps the original encoder-decoder ASR architecture and remaps weights explicitly for MLX inference.
- The example above resamples to 16 kHz before calling `transcribe()`, which matches the runtime requirement.
- This mirror is a duplicated repo, not an automatically synchronized namespace mirror.

## Links

- Canonical MLX repo: [`appautomaton/cohere-asr-mlx`](https://huggingface.co/appautomaton/cohere-asr-mlx)
- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Apache 2.0 — following the upstream Cohere Transcribe model license. Check the original Cohere release for current terms.
