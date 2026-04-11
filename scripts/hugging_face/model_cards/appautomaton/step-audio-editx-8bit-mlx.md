---
library_name: mlx
pipeline_tag: text-to-speech
base_model: stepfun-ai/Step-Audio-EditX
base_model_relation: quantized
license: apache-2.0
language:
- en
- zh
- ja
- ko
tags:
- mlx
- tts
- speech
- voice-cloning
- audio-editing
- step-audio
- step-audio-editx
- stepfun
- quantized
- int8
- apple-silicon
- bundled-components
---

# Step-Audio-EditX — MLX 8-bit

This repository contains a self-contained pure-MLX int8 conversion of
Step-Audio-EditX for local voice cloning and expressive audio editing on
Apple Silicon. All pipeline components are stored as `.safetensors` — no
PyTorch, ONNX, or NumPy files are required at inference time.

## Model Details

- Developed by: AppAutomaton
- Upstream model: [`stepfun-ai/Step-Audio-EditX`](https://huggingface.co/stepfun-ai/Step-Audio-EditX)
- Task: zero-shot voice cloning, expressive audio editing
- Runtime: MLX on Apple Silicon
- Precision: int8 for Step1 LM, Flow model, and VQ02 tokenizer; bf16 for the rest
- Total size: ~4.1 GB (down from ~7.7 GB upstream)

## Bundle Contents

This bundle is self-contained — all weights are packaged in one repository.

| File | Component | Format | Size |
| --- | --- | --- | --- |
| `model.safetensors` | Step1 LM (3.5B params) | int8 | 3.5 GB |
| `flow-model.safetensors` | Flow model (DiT + conformer) | int8 | 181 MB |
| `vq02.safetensors` | VQ02 audio tokenizer | int8 | 162 MB |
| `vq06.safetensors` | VQ06 audio tokenizer | bf16 | 249 MB |
| `hift.safetensors` | HiFT vocoder | bf16 | 40 MB |
| `campplus.safetensors` | CampPlus speaker embedding | bf16 | 13 MB |
| `flow-conditioner.safetensors` | Flow conditioner | bf16 | 2.5 MB |
| `config.json` | Step1 LM config + quantization | JSON | — |
| `flow-model-config.json` | Flow model config | JSON | — |
| `vq02-config.json`, `vq06-config.json` | Tokenizer configs | JSON | — |
| `hift-config.json`, `campplus-config.json`, `flow-conditioner-config.json` | Component configs | JSON | — |
| `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` | Step1 tokenizer | JSON | — |

## How to Get Started

Download the bundle:

```bash
hf download appautomaton/step-audio-editx-8bit-mlx \
  --local-dir models/stepfun/step_audio_editx/mlx-int8
```

**Voice cloning:**

```bash
python scripts/generate/step_audio_editx.py \
  --prompt-audio reference.wav \
  --prompt-text "Transcript of reference audio." \
  -o cloned.wav \
  clone --target-text "New speech in the cloned voice."
```

**Audio editing (change emotion):**

```bash
python scripts/generate/step_audio_editx.py \
  --prompt-audio input.wav \
  --prompt-text "Transcript of input audio." \
  -o happy.wav \
  edit --edit-type emotion --edit-info happy
```

## Supported Edit Types

| Edit type | Description | `--edit-info` examples |
| --- | --- | --- |
| `emotion` | Change the emotion of speech | `happy`, `sad`, `angry`, `surprised` |
| `style` | Change speaking style | `whispering`, `broadcasting`, `formal` |
| `speed` | Change speaking speed | `fast`, `slow` |
| `denoise` | Remove noise from audio | not used |
| `vad` | Remove silences from audio | not used |
| `paralinguistic` | Add non-verbal sounds | requires `--target-text` |

## Architecture

Five-stage pipeline, all running pure MLX with bf16 activations:

1. **Step1 LM** (3.5B params, int8) — autoregressive dual-codebook token generation
2. **CampPlus** (bf16) — speaker embedding extraction from reference audio
3. **Flow conditioner** (bf16) — conditions generation on speaker embedding
4. **Flow model** (int8) — flow-matching mel spectrogram generation
5. **HiFT vocoder** (bf16) — mel spectrogram to waveform

The VQ02 and VQ06 tokenizers encode reference audio into dual codebook tokens
consumed by Step1.

## Performance

On Apple Silicon with int8 weights and bf16 activations, real-time factor
(RTF) is approximately 1.46x for voice cloning — faster than real-time.

## Links

- Source code: [`mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Upstream model: [`stepfun-ai/Step-Audio-EditX`](https://huggingface.co/stepfun-ai/Step-Audio-EditX)
- Technical report: [arXiv:2511.03601](https://arxiv.org/abs/2511.03601)
- More examples: [AppAutomaton](https://github.com/appautomaton)

## License

Apache 2.0 — following the upstream license published with
[`stepfun-ai/Step-Audio-EditX`](https://huggingface.co/stepfun-ai/Step-Audio-EditX).
