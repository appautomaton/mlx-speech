# Qwen3-ASR

Qwen3-ASR is the local MLX runtime path for Qwen3-ASR-1.7B. The v0 target is
offline, single-pass transcription for English, Chinese, and mixed
Chinese/English speech through the shared `mlx_speech.asr` API.

The runtime does not route inference through PyTorch, Transformers, vLLM,
`qwen_asr`, or the upstream Qwen runtime.

## Getting the Model

Pre-converted BF16 MLX weights are published at
[appautomaton/qwen3-asr-1.7b-bf16-mlx](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx).
Quantized builds (`-int8`, `-mxfp8`) are produced via [Conversion](#conversion)
and published under matching repos; see [Quantization](#quantization) for the
alias map. There are three ways to get the weights, in order of preference:

**1. Load by alias (downloads automatically on first use):**

```python
import mlx_speech

asr = mlx_speech.asr.load("qwen3-asr-1.7b")        # bf16 today (int8 after rollout)
asr = mlx_speech.asr.load("qwen3-asr-1.7b-int8")   # affine int8
asr = mlx_speech.asr.load("qwen3-asr-1.7b-mxfp8")  # microscaling FP8
```

```bash
mlx-speech asr \
  --model qwen3-asr-1.7b \
  --audio speech.wav
```

The full repo ID `appautomaton/qwen3-asr-1.7b-bf16-mlx` works in place of the
alias.

**2. Download once into `models/` and load by local path:**

```bash
hf download appautomaton/qwen3-asr-1.7b-bf16-mlx \
  --local-dir models/qwen3_asr_1_7b/mlx-bf16
```

**3. Convert from the upstream checkpoint yourself** — see
[Conversion](#conversion) below.

## Local Layout

```text
models/qwen3_asr_1_7b/
  original/    # upstream BF16 .safetensors (conversion input)
  mlx-bf16/    # unquantized MLX runtime package
  mlx-int8/    # affine int8 (group_size 64)
  mlx-mxfp8/   # microscaling FP8 (group_size 32)
```

The project `models/` entry may be a symlink to a shared local model store —
the loader follows it transparently.

## Conversion

The upstream Qwen files are already BF16 `.safetensors`. The converter renames
checkpoint keys from `thinker.*` into this repo's MLX module tree and transposes
the audio Conv2D weights from PyTorch layout into MLX layout. It can emit an
unquantized bf16 package or a quantized package — see [Quantization](#quantization).

```bash
# default is int8; pass --quant bf16/int8/mxfp8
python scripts/convert/qwen3_asr.py --quant int8
python scripts/convert/qwen3_asr.py --quant mxfp8
python scripts/convert/qwen3_asr.py --quant bf16
```

Output lands in `models/qwen3_asr_1_7b/mlx-<quant>/` by default; pass
`--input-dir` / `--output-dir` to override.

## Quantization

Two 8-bit builds are supported in addition to bf16:

| Build | Mode | group_size | Bias | Alias |
| --- | --- | --- | --- | --- |
| bf16 | — | — | — | `qwen3-asr-1.7b-bf16` |
| int8 | affine | 64 | yes | `qwen3-asr-1.7b-int8` |
| mxfp8 | microscaling FP8 (E4M3 / E8M0 scale) | 32 | no | `qwen3-asr-1.7b-mxfp8` |

The quantization mode is stored in each package's `config.json` (`quantization`
block) and re-applied automatically on load, so it can never desync from the
weights. Selection is alias-based or by explicit path — both work:

```python
asr = mlx_speech.asr.load("qwen3-asr-1.7b-mxfp8")
asr = mlx_speech.asr.load("models/qwen3_asr_1_7b/mlx-mxfp8")
```

mxfp8 requires `group_size=32` and has no bias term (MLX 0.31.1+). Quantization
covers the Linear and Embedding layers across the audio tower and text decoder;
Conv2D layers stay unquantized.

## Quick Start

```python
import mlx_speech

asr = mlx_speech.asr.load("qwen3-asr-1.7b")
result = asr.generate("speech.wav", max_new_tokens=256)
print(result.language, result.text)
```

```bash
mlx-speech asr \
  --model qwen3-asr-1.7b \
  --audio speech.wav
```

A local path such as `models/qwen3_asr_1_7b/mlx-int8` works in place of the
alias everywhere below.

## Language Behavior

The CLI default omits `--language`, which lets Qwen3-ASR infer the language from
the audio. This remains the default mode and is the right first option for
single-language English or Chinese speech.

For Chinese/English mixed speech where preserving Chinese characters matters,
prefer the Chinese prompt path for now:

```python
asr.generate("mixed-speech.wav", language="Chinese")
```

```bash
mlx-speech asr \
  --model models/qwen3_asr_1_7b/mlx-int8 \
  --audio mixed-speech.wav \
  --language Chinese
```

Local smoke tests found that auto mode can treat English-dominant mixed speech
as English and translate Chinese segments into English. The Chinese prompt path
preserved mixed Chinese/English text best in those checks.

Explicit language forcing is available when desired:

```python
asr.generate("speech.wav", language="Chinese")
asr.generate("speech.wav", language="English")
```

```bash
mlx-speech asr \
  --model models/qwen3_asr_1_7b/mlx-int8 \
  --audio speech.wav \
  --language Chinese
```

`--language auto`, `language=None`, and an empty language all use auto-detect
prompting for Qwen3-ASR. Cohere and Granite keep their existing English default
when language is omitted.

## Runtime Shape

- Audio is loaded or expected as 16 kHz mono waveform data.
- The frontend matches the Qwen `WhisperFeatureExtractor` setup: 128 mel bins,
  `n_fft=400`, `hop_length=160`, and dynamic padding.
- The processor builds the Qwen chat prompt directly with token IDs and expands
  `<|audio_pad|>` to the exact audio feature length.
- The MLX runtime validates `prompt_tokens + max_new_tokens` before generation.
- Audio embeddings replace audio placeholder token embeddings before Qwen3
  prefill.
- Generation uses greedy decoding with a local KV cache and parses
  `language ...<asr_text>...` outputs into `(language, text)`.
- Post-processing trims known generated tail markers such as repeated assistant
  labels from the transcript returned by the public API.

## Current Limits

- v0 is single-pass offline ASR only.
- Streaming is deferred.
- Timestamps and forced alignment are deferred.
- Long-audio chunking and language merge logic are deferred.
- Broader multilingual validation beyond English, Chinese, and mixed
  Chinese/English is deferred.
- 8-bit affine int8 and mxfp8 builds are supported (see
  [Quantization](#quantization)); 4-bit (mxfp4/nvfp4) is deferred.

## Reference Source

`.references/Qwen3-ASR` is source-only evidence for processor, prompt, parser,
audio tower, text decoder, and backend behavior. It is not vendored runtime
code and is not imported by `mlx-speech`.
