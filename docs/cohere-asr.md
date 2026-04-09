# CohereASR

## What It Is

`CohereASR` is the transcription family in this repo.

It is the best fit for:

- local speech transcription
- multilingual greedy decode
- chunked long-form audio transcription

Current local default runtime:

- `models/cohere/cohere_transcribe/mlx-int8`

## Main Python API

Re-exported entry point:

- `from mlx_speech.generation import CohereAsrModel`

Loaders:

- `CohereAsrModel.from_dir(...)`
- `CohereAsrModel.from_path(...)`

Main inference methods:

- `transcribe(audio, sample_rate=16000, language="en", punctuation=True, itn=False, max_new_tokens=448)`
- `transcribe_batch(audios, sample_rate=16000, language="en", punctuation=True, itn=False, max_new_tokens=448)`

Return type:

- `CohereAsrResult`
  - `text`
  - `tokens`
  - `language`

## Current CLI

Scripts:

- `scripts/transcribe_cohere_asr.py`
- `scripts/convert_cohere_asr.py`

Transcription flags:

- `--audio`
- `--model-dir`
- `--language`
- `--no-punctuation`
- `--itn`
- `--max-new-tokens`
- `--output`

Conversion flags:

- `--input-dir`
- `--output-dir`
- `--bits`
- `--group-size`
- `--mode`
- `--skip-supporting-files`

## Runtime Reality

What is landed locally now:

- MLX encoder-decoder runtime
- tokenizer + feature extractor loading from the checkpoint directory
- quantized checkpoint loading
- sequential chunk handling for longer audio

## Important Constraints

- `CohereAsrModel.transcribe()` expects `16 kHz` audio
- the CLI resamples input audio to `16 kHz` before calling the model
- decoding is greedy in the current public surface
- `transcribe_batch()` preserves order but runs sequentially

## Example Usage

Python:

```python
import numpy as np
from mlx_speech.generation import CohereAsrModel

audio = np.load("speech.npy").astype(np.float32)
model = CohereAsrModel.from_dir("models/cohere/cohere_transcribe/mlx-int8")
result = model.transcribe(audio, sample_rate=16000, language="en")
print(result.text)
```

CLI:

```bash
python scripts/transcribe_cohere_asr.py \
  --audio speech.wav \
  --language en
```
