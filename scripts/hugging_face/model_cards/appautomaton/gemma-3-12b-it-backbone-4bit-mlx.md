---
library_name: mlx
pipeline_tag: feature-extraction
language:
- multilingual
license: gemma
base_model:
- google/gemma-3-12b-it
base_model_relation: quantized
tags:
- mlx
- gemma3
- gemma
- text-encoder
- feature-extraction
- dramabox
- apple-silicon
- quantized
- 4-bit
---

# Gemma 3 12B IT Text-Encoder Backbone (MLX, 4-bit)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.github.io)
[![DramaBox TTS](https://img.shields.io/badge/%F0%9F%A4%97%20model-DramaBox%20TTS-yellow)](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)

MLX 4-bit conversion of the Gemma 3 12B IT text backbone. It serves as the text-conditioning encoder for [DramaBox TTS](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx) in [mlx-speech](https://github.com/appautomaton/mlx-speech), exposing the per-layer hidden states the audio model conditions on. It is not a chat or text-generation model.

> **Backbone, not a full Gemma.** The language-model head and generation path are removed. It emits hidden states only. Use a standard Gemma checkpoint for chat or text generation.

## Model Details

- Developed by: [App Automaton](https://appautomaton.github.io)
- Upstream model: [`google/gemma-3-12b-it`](https://huggingface.co/google/gemma-3-12b-it), converted from the `gemma-3-12b-it-qat-q4_0` release
- Role: text-encoder backbone. Produces all 49 hidden states consumed by the DramaBox prompt pipeline.
- Quantization: MLX 4-bit affine, `group_size=64`, with bf16 scales and norms
- Runtime: MLX on Apple Silicon

## Contents

| File | Component | Format | Size |
| --- | --- | --- | --- |
| `model-0000{1,2}-of-00002.safetensors` | Gemma 3 12B text backbone | 4-bit affine | ~6.2 GB |
| `model.safetensors.index.json` | Shard index | JSON | n/a |
| `config.json` | Backbone and quantization config | JSON | n/a |
| `tokenizer.json`, `tokenizer.model`, `tokenizer_config.json` | Gemma tokenizer | JSON | n/a |

## How to Get Started

This backbone is the text encoder for DramaBox. Download it alongside the DramaBox weights:

```bash
hf download appautomaton/gemma-3-12b-it-backbone-4bit-mlx \
  --local-dir models/gemma_3_12b_it_backbone/mlx-4bit
```

It is then passed to `DramaBoxModel.from_dir(..., gemma_dir=...)`. See the [DramaBox card](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx) for the full pipeline. To load the backbone directly:

```python
from mlx_speech.models.gemma3_text import load_gemma3_text_model, LTXVGemmaTokenizer

gemma, _ = load_gemma3_text_model("models/gemma_3_12b_it_backbone/mlx-4bit")
tokenizer = LTXVGemmaTokenizer.from_dir("models/gemma_3_12b_it_backbone/mlx-4bit")
```

## Intended Use

Text conditioning for LTX-2-derived audio diffusion models, specifically DramaBox TTS. It is reusable as a Gemma 3 12B feature extractor on Apple Silicon, but it carries no generation head.

## Links

- Source code: [`appautomaton/mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Project page: [appautomaton.github.io/mlx-speech](https://appautomaton.github.io/mlx-speech/)
- Paired model: [`appautomaton/dramabox-tts-3.3b-bf16-mlx`](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)
- More from App Automaton: [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

Gemma. Use is governed by Google's [Gemma Terms of Use](https://ai.google.dev/gemma/terms). By downloading or using these weights you agree to those terms.
