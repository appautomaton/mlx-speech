---
language:
- en
- zh
license: apache-2.0
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: Qwen/Qwen3-ASR-1.7B
base_model_relation: quantized
tags:
- mlx
- asr
- speech-recognition
- transcription
- apple-silicon
- quantized
- 8bit
- int8
- qwen3
---

# Qwen3-ASR-1.7B — MLX (int8)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.github.io)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)
[![bf16 build](https://img.shields.io/badge/variant-bf16-lightgrey)](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx)

MLX-native **int8** conversion of [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) for local automatic speech recognition on Apple Silicon — English, Chinese, and mixed Chinese/English. It runs through the [`mlx-speech`](https://github.com/appautomaton/mlx-speech) runtime with no PyTorch and no cloud API at inference time. Weights ship as plain `.safetensors`.

> **This is the default Qwen3-ASR build in `mlx-speech`.** In local Apple Silicon checks the int8 build produced transcripts **identical to the bf16 build** on our English test clips, while using **~2.3× less peak memory** (≈2.9 GiB vs ≈6.6 GiB) and decoding **~3–4× faster**. For the unquantized reference, see the [bf16 build](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx).

## Model Details

- Developed by: [App Automaton](https://appautomaton.github.io)
- Upstream model: [`Qwen/Qwen3-ASR-1.7B`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) (code: [`QwenLM/Qwen3-ASR`](https://github.com/QwenLM/Qwen3-ASR))
- Task: automatic speech recognition — English, Chinese, and mixed Chinese/English (upstream supports more languages; the `mlx-speech` v0 path validates en/zh/mixed)
- Precision: **int8 affine, group_size 64**, applied to the Linear and Embedding layers across the audio tower and text decoder; Conv2D layers are kept unquantized
- Runtime: MLX on Apple Silicon
- Input: 16 kHz mono audio

## Variants

| Repo | Precision | Notes |
| --- | --- | --- |
| [`qwen3-asr-1.7b-bf16-mlx`](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx) | bf16 | unquantized reference |
| `qwen3-asr-1.7b-int8-mlx` (this repo) | int8 affine | **default** — recommended for most users |

## How to Get Started

Install [`mlx-speech`](https://github.com/appautomaton/mlx-speech), then load by alias (downloads on first use):

```python
import mlx_speech

asr = mlx_speech.asr.load("qwen3-asr-1.7b-int8")
result = asr.generate("speech.wav")               # language=None auto-detects
print(result.language, result.text)

# For mixed Chinese/English, forcing Chinese preserves Chinese characters best:
result = asr.generate("mixed-speech.wav", language="Chinese")
```

```bash
mlx-speech asr --model qwen3-asr-1.7b-int8 --audio speech.wav
```

Or download the weights once and load by local path:

```bash
hf download appautomaton/qwen3-asr-1.7b-int8-mlx \
  --local-dir models/qwen3_asr_1_7b/mlx-int8
```

```python
asr = mlx_speech.asr.load("models/qwen3_asr_1_7b/mlx-int8")
```

## Notes

- This repo contains the quantized MLX runtime artifact only.
- The quantization mode (`affine`, group_size 64, 8-bit) is stored in `config.json` and re-applied automatically on load, so it cannot desync from the weights.
- v0 is single-pass offline ASR; streaming, timestamps, and long-audio chunking are deferred.
- Accuracy was checked relative to the bf16 build on English clips; broader Chinese/mixed evaluation is ongoing.

## Links

- Source code: [`appautomaton/mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Project page: [appautomaton.github.io/mlx-speech](https://appautomaton.github.io/mlx-speech/)
- Unquantized variant: [`appautomaton/qwen3-asr-1.7b-bf16-mlx`](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx)
- More from App Automaton: [Project](https://appautomaton.github.io) · [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

Apache 2.0, following the upstream [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) release. Refer to the original Qwen release for current terms.
