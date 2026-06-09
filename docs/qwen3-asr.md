# Qwen3-ASR

Qwen3-ASR is the local MLX runtime path for Qwen3-ASR-1.7B. The v0 target is
offline, single-pass transcription for English, Chinese, and mixed
Chinese/English speech through the shared `mlx_speech.asr` API.

The runtime is local-path first. It does not download weights automatically and
does not route inference through PyTorch, Transformers, vLLM, `qwen_asr`, or the
upstream Qwen runtime.

## Local Layout

```text
models/Qwen3-ASR-1.7B-MLX-BF16/   # local MLX runtime package
```

The upstream Qwen files are already BF16 `.safetensors`. The converter renames
checkpoint keys from `thinker.*` into this repo's MLX module tree and transposes
the audio Conv2D weights from PyTorch layout into MLX layout. It does not
quantize. The project `models/` entry may be a symlink to a shared local model
store.

```bash
python scripts/convert/qwen3_asr.py \
  --input-dir /path/to/Qwen3-ASR-1.7B-original \
  --output-dir models/Qwen3-ASR-1.7B-MLX-BF16
```

## Quick Start

```python
import mlx_speech

asr = mlx_speech.asr.load("models/Qwen3-ASR-1.7B-MLX-BF16")
result = asr.generate("speech.wav", max_new_tokens=256)
print(result.language, result.text)
```

```bash
mlx-speech asr \
  --model models/Qwen3-ASR-1.7B-MLX-BF16 \
  --audio speech.wav
```

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
  --model models/Qwen3-ASR-1.7B-MLX-BF16 \
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
  --model models/Qwen3-ASR-1.7B-MLX-BF16 \
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
- Automatic model download is deferred; use local paths.
- Broader multilingual validation beyond English, Chinese, and mixed
  Chinese/English is deferred.
- Low-bit quantization is deferred; `Qwen3-ASR-1.7B-MLX-BF16/` is the supported
  package layout.

## Reference Source

`.references/Qwen3-ASR` is source-only evidence for processor, prompt, parser,
audio tower, text decoder, and backend behavior. It is not vendored runtime
code and is not imported by `mlx-speech`.
