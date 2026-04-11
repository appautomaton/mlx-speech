# mlx-speech

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)

> [!NOTE]
> This project wouldn't exist without the inspiration and generous support of the incredible community at [linux.do](https://linux.do).

Local speech synthesis, editing, and transcription on Apple Silicon, running
pure MLX. No cloud, no PyTorch.

| Alias | Type | Description |
| --- | --- | --- |
| `fish-s2-pro` | TTS | Fish S2 Pro — dual-AR TTS, voice cloning, emotion tags |
| `vibevoice` | TTS | VibeVoice Large — hybrid LLM+diffusion TTS, voice cloning |
| `longcat` | TTS | LongCat AudioDiT — flow-matching diffusion TTS |
| `moss-local` | TTS | OpenMOSS TTS Local — local-attention multi-VQ TTS |
| `moss-ttsd` | TTS | OpenMOSS TTS Delay — delay-pattern dialogue TTS |
| `cohere-asr` | ASR | Cohere Transcribe — multilingual ASR |

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.13+

## Installation

```bash
pip install mlx-speech
```

## Quick Start

Models download automatically from [HuggingFace](https://huggingface.co/appautomaton) on first use.

**Python API:**

```python
import mlx_speech

# Text-to-speech
model = mlx_speech.tts.load("fish-s2-pro")
result = model.generate("Hello from mlx-speech!")
# result.waveform: mx.array, result.sample_rate: int

# Voice cloning with emotion tags
result = model.generate(
    "[excited] This is amazing!",
    reference_audio="reference.wav",
    reference_text="Transcript of the reference audio.",
)

# Speech-to-text
asr = mlx_speech.asr.load("cohere-asr")
result = asr.generate("audio.wav")
print(result.text)

# List available models
mlx_speech.tts.list_models()
mlx_speech.asr.list_models()
```

**CLI:**

```bash
# Generate speech
mlx-speech tts --model fish-s2-pro --text "Hello!" -o output.wav

# Voice cloning with emotion tags
mlx-speech tts --model fish-s2-pro \
  --text "[whisper] Just between us..." \
  --reference-audio ref.wav \
  --reference-text "Transcript of reference." \
  -o cloned.wav

# Step Audio emotion editing
mlx-speech tts --model step-audio \
  --reference-audio input.wav \
  --reference-text "Transcript." \
  --edit-type emotion --edit-info happy \
  -o happy.wav

# Sound effect generation
mlx-speech tts --model moss-sound-effect \
  --text "rolling thunder with rainfall" \
  --duration-seconds 8 \
  -o thunder.wav

# Transcribe audio
mlx-speech asr --model cohere-asr --audio speech.wav

# Discover models
mlx-speech tts --list-models
mlx-speech asr --list-models
mlx-speech --help
```

**Local model paths work too:**

```bash
mlx-speech tts --model models/fish_s2_pro/mlx-int8 --text "Hello!" -o output.wav
```

## Models

Pre-converted MLX weights are on Hugging Face under [appautomaton](https://huggingface.co/appautomaton).
Use `mlx_speech.tts.load("alias")` or `mlx_speech.tts.load("appautomaton/repo-name")` to load them.

| Alias | HF Repo | Quant |
| --- | --- | --- |
| `fish-s2-pro` | [fishaudio-s2-pro-8bit-mlx](https://huggingface.co/appautomaton/fishaudio-s2-pro-8bit-mlx) | int8 |
| `vibevoice` | [vibevoice-mlx](https://huggingface.co/appautomaton/vibevoice-mlx) | int8 |
| `longcat` | [longcat-audiodit-3.5b-8bit-mlx](https://huggingface.co/appautomaton/longcat-audiodit-3.5b-8bit-mlx) | int8 |
| `moss-local` | [openmoss-tts-local-mlx](https://huggingface.co/appautomaton/openmoss-tts-local-mlx) | int8 |
| `moss-ttsd` | [openmoss-ttsd-mlx](https://huggingface.co/appautomaton/openmoss-ttsd-mlx) | int8 |
| `moss-sound-effect` | [openmoss-sound-effect-mlx](https://huggingface.co/appautomaton/openmoss-sound-effect-mlx) | 4-bit |
| `step-audio` | [step-audio-editx-8bit-mlx](https://huggingface.co/appautomaton/step-audio-editx-8bit-mlx) | int8 |
| `cohere-asr` | [cohere-asr-mlx](https://huggingface.co/appautomaton/cohere-asr-mlx) | int8 |

## Conversion

Convert from upstream source weights:

```bash
python scripts/convert_fish_s2_pro.py
python scripts/convert_longcat_audiodit.py
python scripts/convert_vibevoice.py
python scripts/convert_moss_local.py
python scripts/convert_moss_ttsd.py
python scripts/convert_cohere_asr.py
```

## Model Guides

Each family has a doc covering behavior, flags, and known limitations:

- [Fish S2 Pro](./docs/fish-s2-pro.md)
- [LongCat AudioDiT](./docs/longcat-audiodit.md)
- [MossTTSLocal](./docs/moss-local.md)
- [MOSS-TTSD](./docs/moss-ttsd.md)
- [MOSS-SoundEffect](./docs/moss-sound-effect.md)
- [VibeVoice](./docs/vibevoice.md)
- [Step-Audio-EditX](./docs/step-audio-editx.md)
- [CohereASR](./docs/cohere-asr.md)

## Development

```bash
git clone https://github.com/appautomaton/mlx-speech.git
cd mlx-speech
uv sync
uv run pytest tests/unit/
uv run ruff check .
```

```text
mlx-speech/
  src/mlx_speech/    library code
  scripts/           conversion and generation entry points
  models/            local checkpoints (not in git)
  tests/             unit, checkpoint, runtime, integration tests
  docs/              model-family behavior guides
```

## License

MIT — see [LICENSE](LICENSE)
