# Granite Speech ASR

Granite Speech ASR is the local MLX runtime for IBM Granite 4.0 1B Speech.
It supports local-path loading of the original sharded `.safetensors`
checkpoint without routing inference through PyTorch, Transformers, `mlx_lm`,
`mlx_audio`, vLLM, or ONNX.

## Quick Start

```python
import mlx_speech

asr = mlx_speech.asr.load("models/ibm/granite_4_0_1b_speech/original")
result = asr.generate("speech.wav", max_new_tokens=200)
print(result.text)
```

```bash
mlx-speech asr \
  --model models/ibm/granite_4_0_1b_speech/original \
  --audio speech.wav
```

For diagnostic batches:

```bash
python scripts/generate/granite_speech_asr.py \
  --model-dir models/ibm/granite_4_0_1b_speech/original \
  --audio speech.wav
```

Transcripts are written under `outputs/granite_speech_asr/transcripts/`, and
`outputs/granite_speech_asr/summary.json` records the input path, output path,
non-empty status, and error text.

## Runtime Shape

- Audio is loaded or expected as 16 kHz mono waveform data.
- The frontend computes HTK log-mel features, pair-stacks adjacent frames, and
  computes the exact number of `<|audio|>` prompt tokens before generation.
- The encoder, QFormer projector, and Granite causal LM are implemented locally
  in MLX with strict checkpoint-key accounting.
- Generation uses greedy decoding with a bounded KV cache. Requests are rejected
  before prefill when `prompt_tokens + max_new_tokens` exceeds the model context.

## Current Limits

- The checked runtime supports greedy transcription only.
- Published `appautomaton` alias weights are not defined yet; use a local model
  directory containing `config.json`, tokenizer assets, and safetensors shards.
- Runtime smoke asserts the bundled sample transcript contains the expected
  phrase `timothy was a spoiled cat`; broader numerical parity checks remain a
  follow-up for reference-level validation.
