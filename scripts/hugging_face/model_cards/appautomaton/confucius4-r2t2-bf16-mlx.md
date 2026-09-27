---
license: other
license_name: netease-model-use-license-agreement
license_link: LICENSE
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: netease-youdao/Confucius4-R2T2
language:
- zh
- en
- yue
- ar
- de
- fr
- es
- pt
- id
- it
- ko
- ru
- th
- vi
- ja
- tr
- hi
- ms
- nl
- sv
- da
- fi
- pl
- cs
- fil
- fa
- el
- ro
- hu
- mk
tags:
- mlx
- apple-silicon
- asr
- speech-recognition
- streaming-asr
- real-time
- confucius4
- r2t2
---

# Confucius4-R2T2 — MLX (bf16)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.com)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-appautomaton-yellow)](https://huggingface.co/appautomaton)

MLX-native **bf16** conversion of NetEase Youdao's [Confucius4-R2T2](https://huggingface.co/netease-youdao/Confucius4-R2T2), a streaming speech recognition model finetuned from Qwen3-ASR-1.7B. It runs through the [`mlx-speech`](https://github.com/appautomaton/mlx-speech) runtime on Apple Silicon with no PyTorch, vLLM, or cloud API at inference time.

> Any modifications made to the original model in this Derivative Work are not endorsed, warranted, or guaranteed by the original right-holder of the original model, and the original right-holder disclaims all liability related to this Derivative Work.

## Model Details

- Upstream developer: NetEase Youdao ([`netease-youdao/Confucius4-R2T2`](https://github.com/netease-youdao/Confucius4-R2T2))
- MLX conversion and runtime: [App Automaton](https://appautomaton.com)
- Architecture: Qwen3-ASR graph (audio encoder + 1.7B text decoder)
- Precision: bf16. The weights are not quantized; keys are remapped to the MLX module tree and audio Conv2D weights transposed to MLX layout.
- Input: 16 kHz mono audio
- Languages: 30, as declared by upstream

## How to Get Started

```bash
pip install "mlx-speech>=0.5.3"
```

Streaming — feed microphone PCM as it arrives:

```python
import mlx_speech

asr = mlx_speech.asr.load("confucius4-r2t2")
session = asr.stream_session(language="English", chunk_ms=160, lookahead_ms=160)
for pcm_chunk in microphone:          # float32, 16 kHz mono, any length
    update = session.feed(pcm_chunk)
    print(update.committed)           # append-only committed text
print(session.finalize().text)
```

Offline:

```python
result = asr.generate("speech.wav")
print(result.text)
```

Or download once and load by path:

```bash
hf download appautomaton/confucius4-r2t2-bf16-mlx \
  --local-dir models/netease/confucius4_r2t2/mlx-bf16
```

## Streaming

`mlx-speech` implements NetEase's chunk loop: prefix rollback, per-window token budget, and append-only commit. Each window reuses the previous window's mel frames, closed audio-encoder blocks, and decoder KV prefix, and recomputes only what changed.

## Notes

- This repo contains the MLX runtime artifact only (no PyTorch checkpoint).

## License

The model weights are licensed under the **NetEase Youdao Model Use License Agreement**; a copy is included in this repository as [`LICENSE`](LICENSE) (original: [MODEL_LICENSE](https://raw.githubusercontent.com/netease-youdao/Confucius4-R2T2/refs/heads/master/MODEL_LICENSE)). By using these weights you agree to its terms, including the restrictions on large-scale commercial use (section 2.2) and prohibited high-risk use (section 4.2). Downstream redistribution must retain the license and this notice.

The `mlx-speech` runtime is licensed separately under its own terms.
