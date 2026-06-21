---
library_name: mlx
pipeline_tag: text-to-speech
language:
- en
license: other
license_name: ltx-2-community
license_link: LICENSE
base_model:
- ResembleAI/Dramabox
tags:
- mlx
- tts
- text-to-speech
- speech
- diffusion
- flow-matching
- dramabox
- resemble
- apple-silicon
---

# DramaBox TTS (MLX, bf16)

[![GitHub](https://img.shields.io/badge/GitHub-mlx--speech-181717?logo=github&logoColor=white)](https://github.com/appautomaton/mlx-speech)
[![App Automaton](https://img.shields.io/badge/App%20Automaton-project-1f6feb)](https://appautomaton.github.io)
[![Gemma backbone](https://img.shields.io/badge/%F0%9F%A4%97%20backbone-Gemma%203%2012B-yellow)](https://huggingface.co/appautomaton/gemma-3-12b-it-backbone-4bit-mlx)

Pure-MLX conversion of [Resemble AI's DramaBox](https://huggingface.co/ResembleAI/Dramabox), an expressive flow-matching diffusion text-to-speech model. It renders 48 kHz stereo speech on Apple Silicon with no PyTorch at inference time. Weights ship as plain `.safetensors` for the [mlx-speech](https://github.com/appautomaton/mlx-speech) runtime.

## Model Details

- Developed by: [App Automaton](https://appautomaton.github.io)
- Upstream model: [`ResembleAI/Dramabox`](https://huggingface.co/ResembleAI/Dramabox), built on [`Lightricks/LTX-2.3`](https://huggingface.co/Lightricks/LTX-2.3)
- Task: English text-to-speech at 48 kHz stereo
- Architecture: Gemma 3 12B text encoder, flow-matching audio DiT (3.3B), audio VAE, BigVGAN + BWE vocoder
- Precision: bf16. This is an MLX format port; the vocoder runs fp32 compute regardless of storage dtype.
- Runtime: MLX on Apple Silicon

> **Requires a separate text encoder.** DramaBox conditions on a Gemma 3 12B backbone. Download it from the paired repo: [`appautomaton/gemma-3-12b-it-backbone-4bit-mlx`](https://huggingface.co/appautomaton/gemma-3-12b-it-backbone-4bit-mlx).

## Contents

| File | Component | Format | Size |
| --- | --- | --- | --- |
| `dramabox-dit-v1.safetensors` | Flow-matching audio DiT (3.3B, 48 layers) | bf16 | ~6.6 GB |
| `dramabox-audio-components.safetensors` | Audio VAE, BigVGAN/BWE vocoder, connector, aggregate embed | bf16 | ~1.9 GB |
| `config.json` | Architecture and inference defaults | JSON | n/a |
| `assets/silence_latent_frame.pt` | Training metadata, unused at inference | n/a | small |

## How to Get Started

DramaBox runs through the [mlx-speech](https://github.com/appautomaton/mlx-speech) repo and needs both this repo and the Gemma backbone.

```bash
# 1. DramaBox weights
hf download appautomaton/dramabox-tts-3.3b-bf16-mlx \
  --local-dir models/dramabox/mlx-bf16

# 2. Gemma 3 12B text-encoder backbone (paired repo)
hf download appautomaton/gemma-3-12b-it-backbone-4bit-mlx \
  --local-dir models/gemma_3_12b_it_backbone/mlx-4bit
```

```python
from mlx_speech.generation.dramabox import DramaBoxModel

model = DramaBoxModel.from_dir(
    "models/dramabox/mlx-bf16",
    gemma_dir="models/gemma_3_12b_it_backbone/mlx-4bit",
)
result = model.generate(
    'A woman speaks clearly, "The weather today will be sunny."',
    duration_s=5.0,
    cfg_scale=2.5,
)
# result.waveform : mx.array [2, T_samples], result.sample_rate : 48000
```

```bash
python scripts/generate_dramabox.py \
  --dramabox-dir models/dramabox/mlx-bf16 \
  --gemma-dir models/gemma_3_12b_it_backbone/mlx-4bit \
  --prompt 'A woman speaks clearly.' \
  --duration 5.0 \
  --out outputs/dramabox.wav
```

## Status and Limitations

- **Works today:** English text to 48 kHz stereo waveform, end to end, pure MLX.
- **Voice-reference cloning:** wired. Pass `voice_ref=` to condition on a speaker; the waveform→mel front-end and the appended reference latent are fully implemented.
- **Spatio-Temporal Guidance:** on by default (`stg_scale=1.5`, block 29), matching the warm-server reference. Set `stg_scale=0` for CFG-only.
- **Optional reference denoising:** set `denoise_ref=True` to clean the voice reference with the [RE-USE / SEMamba enhancer](https://huggingface.co/appautomaton/re-use-semamba-mlx) before conditioning (non-commercial weights; default off).
- **Memory:** the DiT, audio components, and Gemma backbone target a 32 GB Apple Silicon machine.

## Links

- Source code: [`appautomaton/mlx-speech`](https://github.com/appautomaton/mlx-speech)
- Paired text encoder: [`appautomaton/gemma-3-12b-it-backbone-4bit-mlx`](https://huggingface.co/appautomaton/gemma-3-12b-it-backbone-4bit-mlx)
- Optional voice-ref denoiser: [`appautomaton/re-use-semamba-mlx`](https://huggingface.co/appautomaton/re-use-semamba-mlx)
- More from App Automaton: [GitHub](https://github.com/appautomaton) · [Hugging Face](https://huggingface.co/appautomaton)

## License

DramaBox derives from LTX-2.3 and is distributed under the **LTX-2 Community License** (see [`LICENSE`](LICENSE) in this repo). Use is also subject to the terms of the upstream [`ResembleAI/Dramabox`](https://huggingface.co/ResembleAI/Dramabox) release.
