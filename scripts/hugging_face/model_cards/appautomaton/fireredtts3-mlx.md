---
language:
- ar
- yue
- zh
- cs
- nl
- en
- fi
- fr
- de
- el
- hi
- id
- it
- ja
- ko
- pl
- pt
- ro
- ru
- es
- th
- tr
- uk
- vi
license: apache-2.0
library_name: mlx
pipeline_tag: text-to-speech
base_model: FireRedTeam/FireRedTTS3
tags:
- mlx
- mlx-speech
- apple-silicon
- text-to-speech
- voice-cloning
- multilingual
- fireredtts3
- bf16
---

<div align="center">

# FireRedTTS3 for MLX

**Multilingual voice cloning on Apple Silicon**

Base BF16 · Pure MLX inference · Mono 24 kHz audio

[Runtime guide](https://github.com/appautomaton/mlx-speech/blob/main/docs/fireredtts3.md) · [Source code](https://github.com/appautomaton/mlx-speech) · [Upstream model](https://huggingface.co/FireRedTeam/FireRedTTS3) · [Project website](https://appautomaton.renocrypt.com/mlx-speech/)

</div>

This repository brings
[FireRedTTS3 Base](https://huggingface.co/FireRedTeam/FireRedTTS3) to
[mlx-speech](https://github.com/appautomaton/mlx-speech) as a complete MLX
voice-cloning pipeline. Give it a reference recording, its transcript, and the
text you want spoken. It returns a waveform in the reference speaker's voice.

[App Automaton](https://appautomaton.renocrypt.com/) maintains the MLX conversion
and runtime. Speech generation runs locally on your Mac, including speaker
conditioning and waveform reconstruction.

The current release is **Base BF16**, stored in `base/mlx-bf16/`. Each model
variant has its own complete inference bundle. Instruct will be added under
`instruct/` after its MLX pipeline is validated, keeping the Base path stable.

## Start with a reference voice

Requires an Apple Silicon Mac and Python 3.13 or later. Install the current
`mlx-speech` runtime from GitHub:

```bash
pip install "git+https://github.com/appautomaton/mlx-speech.git"
```

The loader downloads the Base bundle on first use. Replace `reference.wav`
and `reference_text` with your recording and its exact transcript. A reference
in the target language is preferred when available.

```python
from mlx_speech import tts
from mlx_speech.audio import write_wav

model = tts.load(
    "appautomaton/fireredtts3-mlx",
    artifact_subdir="base/mlx-bf16",
)
result = model.generate(
    "你好，很高兴认识你。",
    reference_audio="reference.wav",
    reference_text="For Timothy was a spoiled cat, and he allowed no one.",
    language="Chinese",
    seed=1234,
    flow_steps=10,
    guidance_scale=2.0,
)
write_wav("generated.wav", result.waveform, sample_rate=result.sample_rate)
```

The aliases `fireredtts3-base` and `fireredtts3-base-bf16` select the same
bundle. Only `base/mlx-bf16/` and the root model card are downloaded. Additional
variants in this repository do not increase the size of a Base download.

<details>
<summary>The same request from the command line</summary>

```bash
mlx-speech tts \
  --model appautomaton/fireredtts3-mlx \
  --artifact-subdir base/mlx-bf16 \
  --text "你好，很高兴认识你。" \
  --reference-audio reference.wav \
  --reference-text "For Timothy was a spoiled cat, and he allowed no one." \
  --language Chinese \
  --seed 1234 \
  --flow-steps 10 \
  --guidance-scale 2.0 \
  --output generated.wav
```

</details>

## One bundle, complete speech output

The model directory contains all three components and the tokenizer. No
separate codec or speaker-model download is needed.

| Component | Weight file | Size |
| --- | --- | ---: |
| Qwen3 and DiT speech-generation core | `core.safetensors` | 3.950 GiB |
| RedAE waveform encoder and decoder | `redae.safetensors` | 1.758 GiB |
| CAM++ speaker encoder | `speaker.safetensors` | 0.014 GiB |

`config.json`, `tokenizer.json`, `tokenizer_config.json`, and `vocab.json` sit
beside the weight files at the directory root. The three weight files total
**5.722 GiB**. Inference also needs memory for activations and the KV cache.

<details>
<summary>Repository layout</summary>

```text
README.md
LICENSE
base/
  mlx-bf16/
    config.json
    core.safetensors
    redae.safetensors
    speaker.safetensors
    tokenizer.json
    tokenizer_config.json
    vocab.json
```

A downloaded `base/mlx-bf16/` directory also loads directly by local path.
Base is the only model variant included in this release.

</details>

The conversion casts the original FP32 trainable weights to **BF16** and maps
their names and layouts for MLX. RedAE's ISTFT window and CAM++ running
statistics remain FP32. This artifact uses 16-bit floating-point weights and
does not apply INT8 or INT4 quantization.

## Languages

The tokenizer accepts the upstream model's 24 language identifiers and 21
Chinese dialect tags. Pass an explicit language such as `"English"`,
`"Chinese"`, or `"Japanese"` with each request.

<details>
<summary>Accepted language identifiers</summary>

Arabic, Cantonese, Chinese, Czech, Dutch, English, Finnish, French, German,
Greek, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese,
Romanian, Russian, Spanish, Thai, Turkish, Ukrainian, and Vietnamese.

The dialect identifiers use the upstream `ZH_` prefix, for example
`"ZH_Sichuan"` and `"ZH_Shanghai"`. The complete list is defined in the
[MLX tokenizer](https://github.com/appautomaton/mlx-speech/blob/main/src/mlx_speech/models/fireredtts3/tokenizer.py).

</details>

Language coverage comes from the upstream model and tokenizer. The local
quality check below covers one Mandarin request with an English reference.

## Measured locally

The optimized runtime uses bounded RedAE sliding attention, cached Qwen3
continuation, and a compiled DiT tensor region. On the fixed request documented
in the runtime guide, one warmup was excluded before three measured runs.

| Measurement | Result |
| --- | --- |
| Median core-generation time | 2.052 seconds, down from 2.648 seconds |
| Core-generation improvement | 22.5% against the corrected attention baseline |
| MLX peak memory | 5.768 GiB, effectively unchanged |
| Output | Finite, non-silent, mono 24 kHz waveform |
| Seed repeatability | Two bitwise-identical waveforms on one loaded model |
| Local ASR transcript | `你好，很高兴认识你。` |
| CAM++ reference/output cosine | 0.7374 |

Core-generation timing excludes model loading, reference preparation, and
waveform decoding. These measurements describe the local fixture and machine.
They do not establish performance or voice quality across other languages and
recordings. The
[runtime guide](https://github.com/appautomaton/mlx-speech/blob/main/docs/fireredtts3.md#mlx-runtime-behavior)
records the measurement context and a separate long-form comparison.

## Current limits

- Base voice cloning requires both reference audio and its transcript.
  Instruct voice design and audio editing are outside this bundle.
- Generation handles one prepared utterance at a time, with batch size one
  and no waveform streaming.
- Supply the intended spoken form of numbers and abbreviations. Text
  normalization and automatic language detection are caller responsibilities.
- Long-form splitting and joining belong to the application. A local
  long-form test produced severe noise late in both the MLX output and the
  upstream reference, so long-passage quality remains a limitation.

## Provenance and regression coverage

The conversion uses
[FireRedTeam's weights at `dcf1bdcd`](https://huggingface.co/FireRedTeam/FireRedTTS3/tree/dcf1bdcd1b8b25b382fa84c3e34eb82e3054a610)
and follows the
[official implementation at `1d32ba78`](https://github.com/FireRedTeam/FireRedTTS3/tree/1d32ba780da6af37a71bdfd9c68c12003e908a46).
Both revisions are recorded in `config.json`.

The repository also retains small, deterministic golden fixtures for RedAE,
CAM++, DiT, and autoregressive generation. They let maintainers check numerical
regressions without keeping or downloading the full checkpoint. The
[fixture manifest](https://github.com/appautomaton/mlx-speech/blob/main/tests/fixtures/fireredtts3/manifest.json)
records the BF16 bundle's SHA-256 hashes and capture provenance.
Capture and replay use an explicit MLX CPU stream to keep these checks
consistent between developer Macs and CI.

These fixtures exercise tiny models with reproducible synthetic weights.
Checkpoint compatibility and full-model voice quality are covered by separate
tests that require the real weights. The
[fixture guide](https://github.com/appautomaton/mlx-speech/blob/main/docs/fireredtts3.md#checkpoint-independent-regression-vectors)
documents the capture script and regression command.

## License and attribution

FireRedTTS3 is developed by the FireRed Team and released under
[Apache 2.0](https://github.com/FireRedTeam/FireRedTTS3/blob/1d32ba780da6af37a71bdfd9c68c12003e908a46/LICENSE).
The MLX runtime and conversion code are maintained by App Automaton under the
[MIT license](https://github.com/appautomaton/mlx-speech/blob/main/LICENSE).

The upstream project describes voice cloning as intended for academic
research. Use reference recordings with the speaker's permission and identify
synthetic speech clearly. See the
[upstream model card](https://huggingface.co/FireRedTeam/FireRedTTS3#usage-disclaimer)
for its intended-use statement.
