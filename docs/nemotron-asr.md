# Nemotron 3.5 ASR Streaming

Nemotron 3.5 ASR is a 0.6B multilingual FastConformer-RNN-T model with native
cache-aware streaming. This port runs the complete inference path in MLX:
waveform to log-mel features, causal subsampling, 24 encoder blocks, language
prompt fusion, and greedy RNN-T decoding.

The current package is local-path first:

```text
models/nvidia/nemotron_3_5_asr_streaming_0_6b/
  original/   # extracted NVIDIA .nemo checkpoint
  mlx-bf16/   # converted MLX runtime package
```

Published bf16 and int8 aliases are added only after their release gate.

## Load and transcribe

```python
import mlx_speech

model = mlx_speech.asr.load(
    "models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16"
)
result = model.generate(
    "speech.wav",
    language="en-US",
    att_context_size=(56, 13),
)
print(result.language, result.text)
```

Audio arrays must be mono 16 kHz float samples. File inputs are loaded,
mixed down, and resampled by the public adapter. Inference is batch-one and
greedy; beam search is not implemented.

## Live session

```python
session = model.stream_session(
    language="en-US",
    att_context_size=(56, 3),
)

for pcm in microphone:
    for token_id in session.feed(pcm):
        consume(token_id)

for token_id in session.finalize():
    consume(token_id)

print(session.result().text)
```

`feed()` accepts arbitrary sample counts, including fewer than the 160-sample
mel hop. The session retains residual PCM, bounded mel history, all 24 attention
and convolution caches, and the RNN-T predictor state. `finalize()` is required:
it supplies the centered-STFT right padding and flushes pending subsampling and
encoder frames.

## The two independent controls

- `att_context_size` selects the model's trained lookahead/accuracy point. Left
  context remains 56 encoder frames (4.48 seconds); right context selects the
  native internal chunk.
- The PCM packet size passed to `feed()` is an application transport choice. It
  can affect callback cadence and Python overhead, but not model context,
  transcript tokens, or accuracy. One feed and arbitrarily ragged feeds are
  tested to produce identical output.

One encoder frame represents 80 ms. NVIDIA's documented modes are:

| `att_context_size` | Native frames | Algorithmic latency | Tier-1 WER |
| --- | ---: | ---: | ---: |
| `(56, 0)` | 1 | 80 ms | 10.38 |
| `(56, 1)` | 2 | 160 ms | 10.00 |
| `(56, 3)` | 4 | 320 ms | 9.49 |
| `(56, 6)` | 7 | 560 ms | 9.12 |
| `(56, 13)` | 14 | 1.12 s | 8.84 |

`(56, 13)` is the offline/default accuracy setting. Smaller right contexts trade
accuracy for earlier encoder output; no retraining is needed.

## Language prompts and quality tiers

Pass a locale such as `en-US`, `de-DE`, or `ja-JP`. `language="auto"` asks the
model to detect the language and emit a terminal language tag; the public result
reports the detected locale. An explicit locale is preferable when known:
NVIDIA reports an average auto-detection accuracy cost, especially on some
non-Latin scripts.

The checkpoint recognizes 40 locales, but they are not one uniform quality tier:

| Tier | Locales |
| --- | --- |
| Transcription-ready (19) | en-US, en-GB, es-US, es-ES, fr-FR, fr-CA, it-IT, pt-BR, pt-PT, nl-NL, de-DE, tr-TR, ru-RU, ar-AR, hi-IN, ja-JP, ko-KR, vi-VN, uk-UA |
| Broad-coverage (13) | pl-PL, sv-SE, cs-CZ, nb-NO, da-DK, bg-BG, fi-FI, hr-HR, sk-SK, zh-CN, hu-HU, ro-RO, et-EE |
| Adaptation-ready (8) | el-GR, lt-LT, lv-LV, mt-MT, sl-SI, he-IL, th-TH, nn-NO |

The final eight are tokenizer-recognized adaptation targets, not production-ready
zero-shot transcription claims.

## Conversion and runtime purity

```bash
uv run --with torch python scripts/convert/nemotron_asr.py \
  --input-dir models/nvidia/nemotron_3_5_asr_streaming_0_6b/original \
  --output-dir models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16
```

PyTorch is confined to this offline `.ckpt` reader and is not a project/runtime
dependency. The converted checkpoint is `.safetensors`; inference imports no
torch, NeMo, or Transformers code.

## Performance

The encoder processes every emitted frame once through every block; it does not
re-encode overlapping windows. Fixed cache storage remains bounded with utterance
length. See the reproducible [Apple M5 Max benchmark](./benchmarks/nemotron-asr-streaming-2026-07-27.md).

## Current limits

- batch size one
- greedy RNN-T only
- no word timestamps or forced alignment
- bf16 is the validated reference build; int8 validation and publication are a
  separate release gate
