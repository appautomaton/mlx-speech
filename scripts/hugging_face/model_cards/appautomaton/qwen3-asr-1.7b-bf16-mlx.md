---
language:
- en
- zh
license: apache-2.0
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: Qwen/Qwen3-ASR-1.7B
tags:
- mlx
- asr
- speech-recognition
- transcription
- apple-silicon
- qwen3
---

# Qwen3-ASR-1.7B — MLX (bf16)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.github.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)
[![int8 build](https://img.shields.io/badge/variant-int8%20(default)-1f6feb)](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx)

MLX-native **bf16** conversion of [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) for local automatic speech recognition on Apple Silicon — English, Chinese, and mixed Chinese/English. It runs through the [`mlx-speech`](https://github.com/appautomaton/mlx-speech) runtime with no PyTorch and no cloud API at inference time. Weights ship as plain `.safetensors`.

> This is the **unquantized reference** build. Most users should prefer the [**int8 build**](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) — in local checks it matched this build's transcripts on English clips while using ~2.3× less memory and decoding ~3–4× faster. Use bf16 when you want the full-precision baseline (e.g. to compare quantization quality yourself).

## Model Details

- Developed by: [App Automaton](https://appautomaton.github.io)
- Upstream model: [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (code: [`QwenLM/Qwen3-ASR`](https://github.com/QwenLM/Qwen3-ASR))
- Task: automatic speech recognition — English, Chinese, and mixed Chinese/English (upstream supports more languages; the `mlx-speech` v0 path validates en/zh/mixed)
- Precision: bf16 — MLX format port; keys are remapped to the MLX module tree and audio Conv2D weights transposed to MLX layout
- Runtime: MLX on Apple Silicon
- Input: 16 kHz mono audio

## Variants

| Repo | Precision | Notes |
| --- | --- | --- |
| `qwen3-asr-1.7b-bf16-mlx` (this repo) | bf16 | unquantized reference |
| [`qwen3-asr-1.7b-int8-mlx`](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) | int8 affine | **default** — recommended for most users |

## How to Get Started

Install [`mlx-speech`](https://github.com/appautomaton/mlx-speech), then load by alias (downloads on first use):

```python
import mlx_speech

asr = mlx_speech.asr.load("qwen3-asr-1.7b-bf16")
result = asr.generate("speech.wav")               # language=None auto-detects
print(result.language, result.text)

# For mixed Chinese/English, forcing Chinese preserves Chinese characters best:
result = asr.generate("mixed-speech.wav", language="Chinese")
```

```bash
mlx-speech asr --model qwen3-asr-1.7b-bf16 --audio speech.wav
```

Or download the weights once and load by local path:

```bash
hf download appautomaton/qwen3-asr-1.7b-bf16-mlx \
  --local-dir models/qwen3_asr_1_7b/mlx-bf16
```

```python
asr = mlx_speech.asr.load("models/qwen3_asr_1_7b/mlx-bf16")
```

## Notes

- This repo contains the MLX runtime artifact only (no PyTorch checkpoint).
- v0 is single-pass offline ASR; streaming, timestamps, and long-audio chunking are deferred.
- For smaller/faster local inference, use the [int8 build](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx).

## Links

- Source code: [`appautomaton/mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Project page: [appautomaton.github.io/mlx-speech](https://appautomaton.github.io/mlx-speech/)
- Default int8 variant: [`appautomaton/qwen3-asr-1.7b-int8-mlx`](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx)
- More from App Automaton: [Project](https://appautomaton.github.io) · [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

Apache 2.0, following the upstream [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) release. Refer to the original Qwen release for current terms.
