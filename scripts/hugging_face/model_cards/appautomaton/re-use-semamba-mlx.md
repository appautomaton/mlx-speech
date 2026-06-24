---
library_name: mlx
pipeline_tag: audio-to-audio
language:
- en
license: other
license_name: nvidia-source-code-license-nc
license_link: https://huggingface.co/nvidia/RE-USE
base_model:
- nvidia/RE-USE
tags:
- mlx
- semamba
- speech-enhancement
- denoising
- mamba
- ssm
- dramabox
- apple-silicon
- non-commercial
---

# RE-USE SEMamba Speech Enhancement (MLX)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.github.io)
[![DramaBox TTS](https://img.shields.io/badge/%F0%9F%A4%97%20model-DramaBox%20TTS-yellow)](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)

Pure-MLX conversion of [NVIDIA RE-USE](https://huggingface.co/nvidia/RE-USE), a
~9.6M-parameter SEMamba universal speech-enhancement model. In
[mlx-speech](https://github.com/appautomaton/mlx-speech) it cleans a voice
reference before VAE conditioning when [DramaBox TTS](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)
runs with `denoise_ref=True`, giving the cloning model a clean speaker anchor.

> **Non-commercial weights.** These weights derive from NVIDIA RE-USE, licensed
> under the NVIDIA Source Code License (non-commercial). See the License section.

## Model Details

- Developed by: [App Automaton](https://appautomaton.github.io)
- Upstream model: [`nvidia/RE-USE`](https://huggingface.co/nvidia/RE-USE) (SEMamba, bidirectional Mamba over STFT magnitude + phase)
- Role: input-side voice-reference denoiser for DramaBox `denoise_ref=True`. Optional, off by default.
- Conversion: format-only port of the fp32 weights to MLX `.safetensors` (1416 keys, ~9.6M params). No quantization, no architecture change.
- Runtime: pure MLX on Apple Silicon. The selective scan mirrors the `mamba_ssm` `selective_scan_ref` reference math, so no CUDA kernels (`mamba-ssm` / `causal-conv1d`) are required.
- Parity: the MLX port matches the torch reference at amplitude-weighted complex correlation 0.9998 (model) and 0.9997 (end-to-end waveform on real speech).

## Contents

| File | Component | Format | Size |
| --- | --- | --- | --- |
| `model.safetensors` | SEMamba enhancer | fp32 | ~38 MB |
| `config.json` | Model + STFT config | JSON | n/a |

## How to Get Started

Used automatically by DramaBox when you opt in:

```python
import mlx_speech

tts = mlx_speech.tts.load("dramabox")
result = tts.generate(
    "Voice cloning from a noisy reference.",
    reference_audio="noisy_speaker.wav",
    denoise_ref=True,   # cleans the reference with this model first
)
```

`tts.load("dramabox")` resolves these weights automatically. To run the enhancer
directly:

```bash
hf download appautomaton/re-use-semamba-mlx --local-dir models/reuse/mlx
```

```python
from pathlib import Path
from mlx_speech.generation.reuse import REUSEEnhancer

enhancer = REUSEEnhancer.from_dir(Path("models/reuse/mlx"))
clean = enhancer.enhance(noisy_waveform, in_sr=16000)  # mono in, mono out
```

## Intended Use

Denoising a short voice-reference clip before voice cloning, so the model
conditions on a clean speaker/style anchor rather than the recording's noise.
The enhancer runs on the reference input, never on generated output, so the
TTS model's paralinguistic events (breaths, laughs) are preserved.

## Links

- Source code: [`appautomaton/mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Project page: [appautomaton.github.io/mlx-speech](https://appautomaton.github.io/mlx-speech/)
- Paired model: [`appautomaton/dramabox-tts-3.3b-bf16-mlx`](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)
- More from App Automaton: [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

NVIDIA Source Code License (non-commercial). These weights are a format
conversion of [`nvidia/RE-USE`](https://huggingface.co/nvidia/RE-USE) and remain
governed by NVIDIA's license terms; by downloading or using them you agree to
those terms. They may not be used commercially. Set `denoise_ref=False` (the
default) to run DramaBox voice cloning without this model. The mlx-speech
runtime code is MIT.
