---
language:
- en
- zh
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-speech
base_model:
- dots-studio/dots.tts-soar
- dots-studio/dots.tts-mf
tags:
- mlx
- text-to-speech
- voice-cloning
- apple-silicon
- qwen2.5
- flow-matching
- meanflow
- quantized
- 8bit
---

# dots.tts SOAR and MeanFlow — MLX

This repository contains four self-contained dots.tts inference artifacts for
[`mlx-speech`](https://github.com/appautomaton/mlx-speech). They run the full
text-to-waveform pipeline in MLX on Apple Silicon without a PyTorch,
Transformers, or `mlx-lm` inference dependency.

## Variants

| Path | Alias | Acoustic solver | Stored precision | Size |
| --- | --- | --- | --- | ---: |
| `soar/mlx-base/` | `dots-tts-soar-base` | SOAR flow matching | Mixed BF16/FP32 | 4.557 GiB |
| `soar/mlx-int8/` | `dots-tts-soar`, `dots-tts-soar-int8` | SOAR flow matching | Selective Qwen int8 | 3.210 GiB |
| `mf/mlx-base/` | `dots-tts-mf-base` | MeanFlow | Mixed BF16/FP32 | 4.559 GiB |
| `mf/mlx-int8/` | `dots-tts-mf`, `dots-tts-mf-int8` | MeanFlow | Selective Qwen int8 | 3.212 GiB |

The short aliases select int8 because both quantized artifacts passed the local
release gate described below. Base and int8 artifacts share one repository but
load independently; `mlx-speech` downloads only the selected subtree and this
root model card.

## Architecture and precision

dots.tts uses text scheduling and a Qwen2.5 contextual trunk to generate
continuous latent speech patches autoregressively. A causal semantic encoder
feeds generated patches back into Qwen. SOAR uses flow matching with
classifier-free guidance; MeanFlow uses a distilled solver without a separate
runtime guidance branch. CAM++ supplies reference-speaker conditioning, and a
causal AudioVAE/BigVGAN path produces mono 48 kHz waveform output.

`mlx-base` is not an all-BF16 checkpoint. Its precision policy is:

| Component | Stored precision |
| --- | --- |
| Qwen, EOS, semantic encoder, DiT/MeanFlow, conditioning projections | BF16 |
| AudioVAE encoder, `enc_mi_layer`, `pre_proj` | FP32 |
| AudioVAE decoder, `dec_mi_layer`, `post_proj` | BF16 |
| CAM++ speaker encoder | FP32 |
| Latent mean and variance | FP32 |

`mlx-int8` applies affine 8-bit quantization with group size 64 only to eligible
native `qwen.model.*` Linear and Embedding modules. Packed weights use U32;
their scales and biases remain BF16. Every non-selected path keeps its exact
`mlx-base` dtype. The artifacts are therefore Qwen-selective int8, not
whole-model 8-bit conversions.

## Usage

Install `mlx-speech` on an Apple Silicon Mac, then load an alias:

```python
from mlx_speech import tts
from mlx_speech.audio import write_wav

model = tts.load("dots-tts-soar")
result = model.generate(
    "Today the weather is bright and peaceful.",
    reference_audio="reference.wav",
    reference_text="My name is Samantha. I speak clearly and calmly.",
    language="en",
    max_audio_patches=128,
    seed=42,
)
write_wav("output.wav", result.waveform, sample_rate=result.sample_rate)
```

Passing reference audio and its matching transcript enables continuation
cloning. To use only the CAM++ speaker embedding, omit `reference_text`:

```python
model = tts.load("dots-tts-mf")
result = model.generate(
    "今天的天气晴朗而平静。",
    reference_audio="reference.wav",
    language="zh",
)
```

Important generation controls are `max_audio_patches`, `solver_steps`,
`guidance_scale` (SOAR only), `speaker_scale`, `language`, `seed`, and
`eos_threshold`. See the
[`mlx-speech` dots.tts guide](https://github.com/appautomaton/mlx-speech/blob/main/docs/dots-tts.md)
for the full control and conversion reference.

## Locally reproduced release gate

App Automaton measured all four artifacts with one fixed macOS English voice
and one fixed macOS Mandarin voice. Each voice was tested in continuation and
speaker-only modes with a 128-patch cap, seed 42, and EOS threshold 0.8. Qwen3
ASR measured content error rate and CAM++ measured speaker cosine.

| Variant | Base WER | Int8 WER | Base speaker cosine | Int8 speaker cosine |
| --- | ---: | ---: | ---: | ---: |
| SOAR | 0.0000 | 0.0000 | 0.7992 | 0.8147 |
| MeanFlow | 0.0588 | 0.0588 | 0.7868 | 0.7901 |
| Overall | 0.0294 | 0.0294 | 0.7930 | 0.8024 |

The pass thresholds were absolute WER regression ≤ 0.01 and speaker-cosine
regression ≤ 0.02. Both int8 artifacts passed. Mandarin error rate used Unicode
Han characters as tokens; English used normalized words.

| Artifact | Observed peak |
| --- | ---: |
| `soar/mlx-base` | 8.308 GiB |
| `soar/mlx-int8` | 6.963 GiB |
| `mf/mlx-base` | 8.521 GiB |
| `mf/mlx-int8` | 7.177 GiB |

These results are reproduced MLX release measurements, not upstream benchmark
claims. The corpus is deliberately small and synthetic, so the results do not
establish equal quality across other voices, languages, prompts, seeds, or
machines. Full prompts, per-case transcripts, hashes, and methodology are in
the checked
[`2026-07-30 quantization report`](https://github.com/appautomaton/mlx-speech/blob/main/docs/benchmarks/dots-tts-quant-gate-2026-07-30.md).

## Provenance

| Source | Pinned revision |
| --- | --- |
| Official implementation, `studio-dots-ai/dots.tts` v0.2.1 | `5ed719e3d36f5a3f6d8037ca9a7009d4fd0520ba` |
| Community MLX comparison, `sb1992/dots-tts-mlx` v0.7.0 | `f64479f51a2a9d7093533732cae86e765d8fb96e` |
| SOAR weights, resolved as `dots-studio/dots.tts-soar` | `e3520f75254d0020a0406db31c51a79d00d22d55` |
| MeanFlow weights, resolved as `dots-studio/dots.tts-mf` | `25c53fb462e57087e52237daa5ea30df1c5cc328` |

The original source identifiers retained in artifact metadata are
`rednote-hilab/dots.tts-soar` and `rednote-hilab/dots.tts-mf`; Hugging Face
resolves them to the `dots-studio` repositories above. Original upstream
checkpoints are not included in this MLX repository.

## Limitations

- The `mlx-speech` dots.tts runtime is inference-only and non-streaming.
- Continuous autoregressive history grows with the reference and generated
  sequence. Peak memory can exceed the measurements above for longer inputs or
  larger patch budgets.
- No-reference generation follows the target-only schedule, but its random
  voice was not a quality-supported release-gate mode.
- English and Mandarin passed the local gate. This card does not publish MLX
  quality measurements for other languages.
- Voice identity and pronunciation depend on reference quality, transcript
  accuracy, text, seed, and generation settings.
- Quantization passed the fixed release corpus but is not claimed to be
  lossless or numerically identical to `mlx-base`.
- Upstream CUDA streaming and real-time measurements do not apply to this
  non-streaming MLX implementation. No real-time performance claim is made.

## Responsible use

High-fidelity voice cloning can enable impersonation and deceptive synthetic
speech. Use a voice only with the speaker's authorization. Treat reference
recordings as biometric data, restrict their storage and access, and disclose
AI-generated audio clearly. Do not use these artifacts for fraud,
misinformation, harassment, deceptive attribution, or evasion of consent.
Deployers are responsible for applicable law, abuse monitoring, and safeguards
appropriate to their users and jurisdiction.

## License and attribution

The official dots.tts code and released checkpoints are provided under the
Apache License 2.0. These MLX conversions preserve that attribution and are
distributed under `apache-2.0`; consult the upstream release and included
metadata when redistributing them.

- Official source: [studio-dots-ai/dots.tts](https://github.com/studio-dots-ai/dots.tts)
- SOAR source weights: [dots-studio/dots.tts-soar](https://huggingface.co/dots-studio/dots.tts-soar)
- MeanFlow source weights: [dots-studio/dots.tts-mf](https://huggingface.co/dots-studio/dots.tts-mf)
- MLX runtime and conversion code: [appautomaton/mlx-speech](https://github.com/appautomaton/mlx-speech)
