---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/1-1/
library_name: mlx
pipeline_tag: automatic-speech-recognition
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
base_model_relation: quantized
language:
- en
- es
- de
- fr
- it
- ar
- ja
- ko
- pt
- ru
- hi
- zh
- vi
- he
- nl
- cs
- da
- pl
- 'no'
- sv
- th
- tr
- bg
- el
- et
- fi
- hr
- hu
- lt
- lv
- ro
- sk
- uk
- mt
- sl
tags:
- mlx
- apple-silicon
- automatic-speech-recognition
- streaming-asr
- multilingual
- fastconformer
- rnnt
- quantized
- int8
---

# Nemotron 3.5 ASR Streaming 0.6B — MLX (int8)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)

Pure-MLX int8 conversion of NVIDIA's [Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) for cache-aware multilingual transcription on Apple Silicon. Linear and embedding layers use 8-bit affine quantization with group size 64. The complete waveform-to-text runtime uses MLX; PyTorch, NeMo, and Transformers are not imported during inference.

> **This is the only published Nemotron build and the default in `mlx-speech`.** A temporary pre-release comparison over 44 minutes of ordinary English and Mandarin web audio found negligible accuracy differences from the local bf16 validation reference: +0.035 percentage points English WER and +0.112 points Mandarin CER.

## Model details

- Upstream developer: NVIDIA
- MLX conversion and runtime: [App Automaton](https://github.com/appautomaton/mlx-speech)
- Architecture: 24-layer cache-aware FastConformer encoder, language prompt, greedy RNN-T decoder
- Parameters: approximately 0.6B
- Precision: int8 affine Linear/Embedding weights, group size 64. Convolutions, normalization, biases, recurrent parameters, unsupported shapes, and other non-eligible tensors retain their native precision; preprocessing buffers remain fp32.
- Input: mono audio, resampled to 16 kHz by the public file adapter
- Runtime: batch one on Apple Silicon

## Usage

```bash
pip install mlx-speech
```

```python
import mlx_speech

asr = mlx_speech.asr.load("nemotron-asr-streaming")
result = asr.generate(
    "speech.wav",
    language="en-US",
    att_context_size=(56, 13),
)
print(result.language, result.text)
```

For a live waveform stream:

```python
session = asr.stream_session(language="en-US", att_context_size=(56, 3))
for pcm_chunk in microphone:
    emitted_token_ids = session.feed(pcm_chunk)
    consume(emitted_token_ids)
session.finalize()
print(session.result().text)
```

`feed()` accepts arbitrary sample counts. `finalize()` is required to flush centered-STFT, subsampling, encoder, and RNN-T state.

## Latency modes

One encoder frame is 80 ms. The left context is 56 frames; right context selects the trained latency/accuracy point without retraining. Tier-1 WER below is reported by NVIDIA for the upstream model.

| Context | Native chunk | Algorithmic latency | Upstream tier-1 WER |
| --- | ---: | ---: | ---: |
| `(56, 0)` | 1 frame | 80 ms | 10.38 |
| `(56, 1)` | 2 frames | 160 ms | 10.00 |
| `(56, 3)` | 4 frames | 320 ms | 9.49 |
| `(56, 6)` | 7 frames | 560 ms | 9.12 |
| `(56, 13)` | 14 frames | 1.12 s | 8.84 |

`(56, 13)` is the default accuracy setting. PCM packet size is independent of model context and does not change the emitted tokens.

## Language tiers

The checkpoint recognizes 40 locales, but they are not one quality tier.

| Tier | Locales |
| --- | --- |
| Transcription-ready (19) | en-US, en-GB, es-US, es-ES, fr-FR, fr-CA, it-IT, pt-BR, pt-PT, nl-NL, de-DE, tr-TR, ru-RU, ar-AR, hi-IN, ja-JP, ko-KR, vi-VN, uk-UA |
| Broad-coverage (13) | pl-PL, sv-SE, cs-CZ, nb-NO, da-DK, bg-BG, fi-FI, hr-HR, sk-SK, zh-CN, hu-HU, ro-RO, et-EE |
| Adaptation-ready (8) | el-GR, lt-LT, lv-LV, mt-MT, sl-SI, he-IL, th-TH, nn-NO |

The adaptation-ready locales are tokenizer-recognized fine-tuning targets, not production-ready zero-shot transcription claims. Use an explicit locale when known; `language="auto"` is supported but can reduce accuracy.

## Pre-release accuracy comparison

Before choosing the release artifact, bf16 and int8 were run at `(56, 13)` on the same four longer recordings with human-provided transcripts. The inputs and detailed hypotheses were kept in a temporary system-cache directory and are not shipped as a benchmark or committed to `mlx-speech`.

| Language | Material | bf16 reference | int8 | Absolute change |
| --- | --- | ---: | ---: | ---: |
| English | Two 15-minute VOA read-along programs ([part 1](https://www.manythings.org/voa/english/201.html), [part 2](https://www.manythings.org/voa/english/202.html)), 2,851 words | 3.788% WER | 3.823% WER | +0.035 pp |
| Mandarin | Two natural 5 Minute Chinese episodes ([mattress](https://5minutechinese.buzzsprout.com/1868166/episodes/19511303-experiencing-chinese-customer-service-through-buying-a-mattress), [mountain](https://5minutechinese.buzzsprout.com/1868166/episodes/19478937-a-chance-encounter-beyond-the-mountain)), 2,679 characters | 5.972% CER | 6.084% CER | +0.112 pp |

The 755,732,373-byte int8 weight file is 40.8% smaller than the local 1,276,192,217-byte bf16 validation artifact. Arbitrarily ragged waveform feeds produce the same tokens as offline int8 inference, and cache storage remains bounded with utterance length.

## Limitations

- batch size one
- greedy RNN-T only; no beam search
- no word timestamps or forced alignment
- inference requires Apple Silicon and MLX
- the pre-release comparison covers English and Mandarin only and is not a substitute for NVIDIA's broader upstream evaluation

## License and attribution

The model weights are governed by **OpenMDW-1.1**. The complete license text is included in this repository as `LICENSE.OpenMDW-1.1`. This is a quantized derivative of [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b); NVIDIA developed and released the upstream model. The `mlx-speech` runtime code is separately licensed under MIT.
