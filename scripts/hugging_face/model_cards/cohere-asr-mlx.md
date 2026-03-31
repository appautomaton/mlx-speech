---
language:
- en
license: cc-by-nc-4.0
library_name: mlx
pipeline_tag: automatic-speech-recognition
tags:
- mlx
- asr
- speech-recognition
- transcription
- apple-silicon
- quantized
- 8bit
---

# Cohere ASR — MLX

Cohere's speech recognition model, converted and quantized for native MLX inference on Apple Silicon.

Fast, accurate transcription running fully locally — no cloud, no API calls.

## Variants

| Path | Precision |
| --- | --- |
| `mlx-int8/` | int8 quantized weights |

## How to Get Started

Via [mlx-speech](https://github.com/appautomaton/mlx-speech):

```bash
python scripts/transcribe_cohere_asr.py \
  --audio input.wav \
  --output transcript.txt
```

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
    audio = np.interp(np.linspace(0, old_len - 1, new_len), np.arange(old_len), audio).astype(np.float32)

model = CohereAsrModel.from_path("mlx-int8")
transcript = model.transcribe(audio, sample_rate=16000, language="en")
```

## Model Details

Converted from the original Cohere ASR checkpoint with explicit MLX weight remapping and int8 quantization. Encoder-decoder architecture optimized for English transcription.

See [mlx-speech](https://github.com/appautomaton/mlx-speech) for the full runtime and conversion code.

## License

cc-by-nc-4.0 — following the upstream Cohere model license. Check the original model release for current terms before commercial use.
