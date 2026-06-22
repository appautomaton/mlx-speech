# mlx-speech

[![PyPI](https://img.shields.io/pypi/v/mlx-speech)](https://pypi.org/project/mlx-speech/)
[![Downloads](https://img.shields.io/pypi/dm/mlx-speech)](https://pypi.org/project/mlx-speech/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20weights-appautomaton-orange)](https://huggingface.co/appautomaton)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/appautomaton/mlx-speech/blob/main/LICENSE)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)

> [!NOTE]
> This project wouldn't exist without the inspiration and generous support of the incredible community at [linux.do](https://linux.do).

Local speech synthesis, editing, and transcription on Apple Silicon, running
pure MLX. No cloud, no PyTorch.

mlx-speech is an [App Automaton](https://appautomaton.github.io) project. The
`appautomaton` org hosts the [code on GitHub](https://github.com/appautomaton/mlx-speech)
and the converted [weights on Hugging Face](https://huggingface.co/appautomaton).

## Models

Pre-converted MLX weights are published under the App Automaton Hugging Face
org, [appautomaton](https://huggingface.co/appautomaton), and download
automatically on first use. Load by alias or by full repo id — `tts.load("fish-s2-pro")` and
`tts.load("appautomaton/fishaudio-s2-pro-8bit-mlx")` are equivalent. Each
model name links to a guide covering behavior, flags, and known limitations.

**Text-to-speech**

| Alias | Model | Weights |
| --- | --- | --- |
| `fish-s2-pro` | [Fish S2 Pro](https://github.com/appautomaton/mlx-speech/blob/main/docs/fish-s2-pro.md) — dual-AR TTS, voice cloning, emotion tags | [int8](https://huggingface.co/appautomaton/fishaudio-s2-pro-8bit-mlx) |
| `vibevoice` | [VibeVoice Large](https://github.com/appautomaton/mlx-speech/blob/main/docs/vibevoice.md) — hybrid LLM+diffusion TTS, voice cloning | [int8](https://huggingface.co/appautomaton/vibevoice-mlx) |
| `longcat` | [LongCat AudioDiT](https://github.com/appautomaton/mlx-speech/blob/main/docs/longcat-audiodit.md) — flow-matching diffusion TTS | [int8](https://huggingface.co/appautomaton/longcat-audiodit-3.5b-8bit-mlx) |
| `moss-local` | [OpenMOSS TTS Local](https://github.com/appautomaton/mlx-speech/blob/main/docs/moss-local.md) — local-attention multi-VQ TTS | [int8](https://huggingface.co/appautomaton/openmoss-tts-local-mlx) |
| `moss-ttsd` | [MOSS-TTSD](https://github.com/appautomaton/mlx-speech/blob/main/docs/moss-ttsd.md) — delay-pattern dialogue TTS | [int8](https://huggingface.co/appautomaton/openmoss-ttsd-mlx) |
| `moss-sound-effect` | [OpenMOSS Sound Effect](https://github.com/appautomaton/mlx-speech/blob/main/docs/moss-sound-effect.md) — text-to-sound-effect generation | [4-bit](https://huggingface.co/appautomaton/openmoss-sound-effect-mlx) |
| `step-audio` | [Step-Audio-EditX](https://github.com/appautomaton/mlx-speech/blob/main/docs/step-audio-editx.md) — voice cloning, audio editing | [int8](https://huggingface.co/appautomaton/step-audio-editx-8bit-mlx) |
| `dramabox` | [DramaBox](https://github.com/appautomaton/mlx-speech/blob/main/docs/dramabox.md) — Resemble flow-matching diffusion TTS, 48 kHz stereo | [bf16](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)¹ |

**Speech-to-text**

| Alias | Model | Weights |
| --- | --- | --- |
| `cohere-asr` | [Cohere Transcribe](https://github.com/appautomaton/mlx-speech/blob/main/docs/cohere-asr.md) — multilingual ASR | [int8](https://huggingface.co/appautomaton/cohere-asr-mlx) |
| `qwen3-asr-1.7b` | [Qwen3-ASR-1.7B](https://github.com/appautomaton/mlx-speech/blob/main/docs/qwen3-asr.md) — English, Chinese, and mixed Chinese/English ASR | [int8](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) · [bf16](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx) |
| — | [IBM Granite Speech 4.0 1B](https://github.com/appautomaton/mlx-speech/blob/main/docs/granite-speech-asr.md) — runs the original sharded checkpoint from a local path | local checkpoint |

¹ `tts.load("dramabox")` also pulls the [Gemma 3 12B backbone](https://huggingface.co/appautomaton/gemma-3-12b-it-backbone-4bit-mlx)
text encoder automatically. Output is 48 kHz stereo. For advanced controls (cfg,
steps, voice reference) use `scripts/generate_dramabox.py`. Optional
`denoise_ref=True` cleans a noisy voice reference with the pure-MLX
[RE-USE / SEMamba enhancer](https://huggingface.co/appautomaton/re-use-semamba-mlx)
(off by default; NSCLv1 non-commercial weights). See
[docs/dramabox.md](https://github.com/appautomaton/mlx-speech/blob/main/docs/dramabox.md).

## Installation

Requires an Apple Silicon Mac (M1 or later) and Python 3.13+.

```bash
pip install mlx-speech
```

## Quick Start

**Python:**

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
asr = mlx_speech.asr.load("qwen3-asr-1.7b")
print(asr.generate("audio.wav").text)

# Local checkpoint paths work anywhere an alias does
granite = mlx_speech.asr.load("models/ibm/granite_4_0_1b_speech/original")
print(granite.generate("audio.wav").text)

# Discover models
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
mlx-speech asr --model qwen3-asr-1.7b --audio speech.wav --language Chinese

# Local checkpoint paths work anywhere an alias does
mlx-speech tts --model models/fish_s2_pro/mlx-int8 --text "Hello!" -o output.wav
mlx-speech asr --model models/ibm/granite_4_0_1b_speech/original --audio speech.wav

# Discover models
mlx-speech tts --list-models
mlx-speech asr --list-models
mlx-speech --help
```

> **Note:** The `mlx-speech` CLI covers the common path — basic generation,
> voice cloning, and editing. For advanced controls (sampling temperature,
> top-p/k, diffusion steps, batch JSONL, duration tuning, etc.) use the
> scripts in `scripts/` directly. Each model family has a corresponding
> script with the full inference surface documented in `docs/`.

## Conversion

To convert upstream source weights yourself:

```bash
python scripts/convert/fish_s2_pro.py
python scripts/convert/longcat_audiodit.py
python scripts/convert/vibevoice.py
python scripts/convert/moss_local.py
python scripts/convert/moss_ttsd.py
python scripts/convert/moss_sound_effect.py
python scripts/convert/step_audio_editx.py
python scripts/convert/cohere_asr.py
python scripts/convert/qwen3_asr.py
```

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
  scripts/           conversion, generation, eval, and audit entry points
  models/            local checkpoints (not in git)
  tests/             unit, checkpoint, runtime, integration tests
  docs/              model-family behavior guides
```

## License

MIT — see [LICENSE](https://github.com/appautomaton/mlx-speech/blob/main/LICENSE)

Built and maintained by [App Automaton](https://appautomaton.github.io).
