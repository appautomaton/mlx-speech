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
  --audio speech.wav \
  --memory-telemetry
```

Transcripts are written under `outputs/granite_speech_asr/transcripts/`, and
`outputs/granite_speech_asr/summary.json` records the input path, output path,
non-empty status, error text, token counts, wall time, and optional coarse MLX
memory snapshots.

For manual long-audio checks, use the `/tmp` benchmark driver. This is not a
default automated-build path because it downloads public-domain media and
requires local IBM checkpoint files:

```bash
tmpdir=$(mktemp -d /tmp/granite-long-audio.XXXXXX)
python scripts/eval/granite_speech_long_audio.py \
  --output-dir "$tmpdir" \
  --source three-bears-catamount \
  --chunk-seconds 120 \
  --max-new-tokens 350
```

The driver keeps source media, chunk WAVs, transcripts, and `summary.json` under
`/tmp`. It reports chunk count, duration, prompt/generated token totals, wall
time, RTF/RTFx, coarse memory snapshots, and normalized word metrics against
the matching Project Gutenberg chapter text.

## Runtime Shape

- Audio is loaded or expected as 16 kHz mono waveform data.
- The frontend computes HTK log-mel features, pair-stacks adjacent frames, and
  computes the exact number of `<|audio|>` prompt tokens before generation.
- The encoder, QFormer projector, and Granite causal LM are implemented locally
  in MLX with strict checkpoint-key accounting.
- Generation uses greedy decoding with a bounded KV cache. Requests are rejected
  before prefill when `prompt_tokens + max_new_tokens` exceeds the model context.
- Attention uses MLX efficient scaled-dot-product attention with grouped-query
  KV heads left unexpanded, avoiding explicit `[heads, tokens, tokens]`
  attention score and weight materialization.
- Context validation happens before STFT, encoder, and projector work when the
  sample count proves a request cannot fit.

## Current Limits

- The checked runtime supports greedy transcription only.
- Published `appautomaton` alias weights are not defined yet; use a local model
  directory containing `config.json`, tokenizer assets, and safetensors shards.
- Ten-minute-plus audio exceeds the model context as a single prompt. Use
  context-safe chunking for long-form checks.
- Runtime smoke asserts the bundled sample transcript contains the expected
  phrase `timothy was a spoiled cat`; broader numerical parity checks remain a
  follow-up for reference-level validation.
