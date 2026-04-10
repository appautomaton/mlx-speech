# LongCat AudioDiT

## What It Is

LongCat AudioDiT is a 3.5B parameter flow-matching diffusion TTS model. It uses a T5 text encoder and a DiT (Diffusion Transformer) backbone with a VAE decoder. It supports zero-shot voice cloning from reference audio and produces high-quality long-form speech.

## Key Properties

| Property | Value |
| --- | --- |
| Architecture | Flow-matching DiT + T5 text encoder + VAE |
| Sample rate | 24,000 Hz |
| Max duration | 60 seconds (hard cap) |
| Quantization | int8 |
| Generation | Non-autoregressive (diffusion) |

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("longcat")
result = model.generate("Hello from LongCat AudioDiT!")
# result.waveform, result.sample_rate
```

```bash
mlx-speech-tts --model longcat --text "Hello!" -o output.wav
```

## Voice Cloning

Provide a reference audio file and its transcript. Both are required.

```python
result = model.generate(
    "This is my cloned voice speaking new text.",
    reference_audio="reference.wav",
    reference_text="Transcript of what is spoken in reference.wav.",
)
```

```bash
mlx-speech-tts --model longcat \
  --text "New speech in the cloned voice." \
  --reference-audio reference.wav \
  --reference-text "Transcript of the reference audio." \
  -o cloned.wav
```

The reference audio is encoded through the VAE to extract speaker characteristics. Longer and cleaner reference clips generally produce better cloning results.

## Generation Parameters

Backend-specific parameters can be passed as keyword arguments:

| Parameter | Default | Notes |
| --- | --- | --- |
| `nfe` | 16 | Number of function evaluations (diffusion steps). Higher = better quality, slower. |
| `guidance_strength` | 4.0 | Classifier-free guidance scale. Higher = more faithful to text, less natural. |

```python
result = model.generate(
    "High quality output with more diffusion steps.",
    nfe=32,
    guidance_strength=5.0,
)
```

## Duration Limits

- **Maximum:** 60 seconds per generation (model constraint)
- **Duration estimation:** The model auto-estimates duration from text length
- **Long text:** If your text exceeds ~60 seconds of speech, split it into segments

For batch generation from a manifest file, use the script:

```bash
python scripts/batch_generate_longcat_audiodit.py --manifest manifest.txt
```

Manifest format (tab-separated):
```
uid	prompt_text	prompt_wav_path	gen_text
```

## Differences from Fish S2 Pro

| | LongCat AudioDiT | Fish S2 Pro |
| --- | --- | --- |
| Architecture | Diffusion (non-autoregressive) | Dual-AR (autoregressive) |
| Sample rate | 24,000 Hz | 44,100 Hz |
| Emotion tags | Not supported | `[excited]`, `[whisper]`, etc. |
| Max duration | 60 seconds (hard cap) | Limited by `max_new_tokens` |
| Generation speed | Fixed cost (NFE steps) | Proportional to output length |
| Quality tradeoff | `nfe` (more steps = better) | `temperature` / `top_p` |

## HuggingFace

- Alias: `longcat`
- Repo: [appautomaton/longcat-audiodit-3.5b-8bit-mlx](https://huggingface.co/appautomaton/longcat-audiodit-3.5b-8bit-mlx)

## Script

```bash
python scripts/generate_longcat_audiodit.py \
  --text "Hello from LongCat." \
  --model-dir models/longcat_audiodit/mlx-int8 \
  --output outputs/longcat.wav
```
