# Granite Speech ASR

Granite Speech ASR is the pure-MLX runtime for IBM Granite 4.0 1B Speech. The
default published artifact applies affine int8 weight quantization to the
Granite causal LM while retaining the acoustic encoder, QFormer projector,
normalization, convolution, biases, activations, and KV cache in BF16. Neither
the original nor quantized path routes inference through PyTorch, Transformers,
`mlx_lm`, `mlx_audio`, vLLM, or ONNX.

## Checkpoint selection

| Selector | Precision | Source |
| --- | --- | --- |
| `granite-speech-4.0-1b` | selective affine int8, group size 64 | `appautomaton/granite-4.0-1b-speech-int8-mlx` |
| `granite-speech-4.0-1b-int8` | explicit alias for the same artifact | `appautomaton/granite-4.0-1b-speech-int8-mlx` |
| local `original/` path | original sharded BF16 | `ibm-granite/granite-4.0-1b-speech` |

The published repository is a single self-contained runtime artifact at its
root. It does not contain the original BF16 shards or an `mlx-int8/` wrapper
directory. Artifact size, memory, speed, and transcript differences are recorded
in the [2026-08-03 int8 quantization gate](benchmarks/granite-speech-int8-quant-gate-2026-08-03.md).

## Quick Start

```python
import mlx_speech

asr = mlx_speech.asr.load("granite-speech-4.0-1b")
result = asr.generate("speech.wav", max_new_tokens=200)
print(result.text)
```

```bash
mlx-speech asr \
  --model granite-speech-4.0-1b \
  --audio speech.wav
```

For diagnostic batches:

```bash
python scripts/generate/granite_speech_asr.py \
  --model-dir models/ibm/granite_4_0_1b_speech/mlx-int8 \
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

## Local conversion

Build the published artifact layout from the original IBM checkpoint:

```bash
python scripts/convert/granite_speech_asr.py
```

The default conversion reads
`models/ibm/granite_4_0_1b_speech/original` and writes
`models/ibm/granite_4_0_1b_speech/mlx-int8`. Quantization metadata is stored in
`config.json`; loading reconstructs the saved quantized module set from its
`.scales` tensors before strict checkpoint alignment.

## Current Limits

- The checked runtime supports greedy transcription only.
- The checked runtime supports upstream speech inputs in English, French,
  German, Spanish, Portuguese, and Japanese. Mandarin speech transcription is
  not an upstream capability.
- Ten-minute-plus audio exceeds the model context as a single prompt. Use
  context-safe chunking for long-form checks.
- Runtime smoke asserts the bundled sample transcript contains the expected
  phrase `timothy was a spoiled cat`; broader numerical parity checks remain a
  follow-up for reference-level validation.
