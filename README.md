<div align="center">

# mlx-speech

**Text-to-speech and speech recognition on Apple Silicon**

Voice cloning, audio editing, sound effects, and transcription. All running locally in MLX.

[![PyPI](https://img.shields.io/pypi/v/mlx-speech)](https://pypi.org/project/mlx-speech/)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/appautomaton/mlx-speech/blob/main/LICENSE)
[![CI](https://github.com/appautomaton/mlx-speech/actions/workflows/ci.yml/badge.svg)](https://github.com/appautomaton/mlx-speech/actions/workflows/ci.yml)

[Quick start](#quick-start) · [Models](#models) · [Model guides](https://github.com/appautomaton/mlx-speech/tree/main/docs) · [Hugging Face weights](https://huggingface.co/appautomaton) · [Project website](https://appautomaton.renocrypt.com/mlx-speech/)

</div>

mlx-speech is an open-source Python library for text-to-speech (TTS) and
automatic speech recognition (ASR) on Apple Silicon Macs. Models share a Python
API and command-line interface, with inference implemented in
[Apple's MLX framework](https://github.com/ml-explore/mlx).

Audio processing stays on your Mac. Inference needs neither PyTorch nor a cloud
service. Published weights download on first use, and local checkpoint paths
support offline loading.

## Installation

Requires an Apple Silicon Mac (M1 or later) and Python 3.13+.

```bash
pip install mlx-speech
```

## Quick Start

**Python:**

```python
import mlx_speech
from mlx_speech.audio import write_wav

# Text-to-speech
model = mlx_speech.tts.load("fish-s2-pro")
result = model.generate("Hello from mlx-speech!")
write_wav("output.wav", result.waveform, sample_rate=result.sample_rate)

# Speech-to-text
asr = mlx_speech.asr.load("qwen3-asr-1.7b")
print(asr.generate("audio.wav").text)
```

**CLI:**

```bash
mlx-speech tts --model fish-s2-pro --text "Hello!" -o output.wav
mlx-speech asr --model qwen3-asr-1.7b --audio speech.wav
```

## Models

Choose a model by task, then use its selector with `tts.load()`, `asr.load()`, or
the CLI's `--model` flag. Model names link to guides with examples, controls,
and limitations. Weight links open the corresponding Hugging Face repositories.

### Text-to-speech, voice cloning, and sound effects

| Model | Use it for | Selector | Weights |
| --- | --- | --- | --- |
| [Fish S2 Pro](https://github.com/appautomaton/mlx-speech/blob/main/docs/fish-s2-pro.md) | Voice cloning and emotion tags | `fish-s2-pro` | [int8](https://huggingface.co/appautomaton/fishaudio-s2-pro-8bit-mlx) |
| [VibeVoice Large](https://github.com/appautomaton/mlx-speech/blob/main/docs/vibevoice.md) | Speech synthesis and voice cloning | `vibevoice` | [int8](https://huggingface.co/appautomaton/vibevoice-mlx) |
| [LongCat AudioDiT](https://github.com/appautomaton/mlx-speech/blob/main/docs/longcat-audiodit.md) | Diffusion speech synthesis | `longcat` | [int8](https://huggingface.co/appautomaton/longcat-audiodit-3.5b-8bit-mlx) |
| [OpenMOSS TTS Local](https://github.com/appautomaton/mlx-speech/blob/main/docs/moss-local.md) | Speech synthesis and voice cloning | `moss-local` | [int8](https://huggingface.co/appautomaton/openmoss-tts-local-mlx) |
| [MOSS-TTSD](https://github.com/appautomaton/mlx-speech/blob/main/docs/moss-ttsd.md) | Multi-speaker dialogue | `moss-ttsd` | [int8](https://huggingface.co/appautomaton/openmoss-ttsd-mlx) |
| [OpenMOSS Sound Effect](https://github.com/appautomaton/mlx-speech/blob/main/docs/moss-sound-effect.md) | Sound effects from text | `moss-sound-effect` | [4-bit](https://huggingface.co/appautomaton/openmoss-sound-effect-mlx) |
| [Step-Audio-EditX](https://github.com/appautomaton/mlx-speech/blob/main/docs/step-audio-editx.md) | Voice cloning and audio editing | `step-audio` | [int8](https://huggingface.co/appautomaton/step-audio-editx-8bit-mlx) |
| [DramaBox](https://github.com/appautomaton/mlx-speech/blob/main/docs/dramabox.md) | Speech synthesis in 48 kHz stereo | `dramabox` | [BF16](https://huggingface.co/appautomaton/dramabox-tts-3.3b-bf16-mlx)¹ |
| [dots.tts SOAR](https://github.com/appautomaton/mlx-speech/blob/main/docs/dots-tts.md) | Voice cloning and waveform streaming | `dots-tts-soar` | [int8 + base](https://huggingface.co/appautomaton/dots-tts-mlx) |
| [dots.tts MeanFlow](https://github.com/appautomaton/mlx-speech/blob/main/docs/dots-tts.md) | Distilled TTS and waveform streaming | `dots-tts-mf` | [int8 + base](https://huggingface.co/appautomaton/dots-tts-mlx) |
| [FireRedTTS3 Base](https://github.com/appautomaton/mlx-speech/blob/main/docs/fireredtts3.md) | Multilingual voice cloning at 24 kHz | `fireredtts3-base` | [BF16](https://huggingface.co/appautomaton/fireredtts3-mlx/tree/main/base/mlx-bf16) |

### Speech-to-text

| Model | Use it for | Selector | Weights |
| --- | --- | --- | --- |
| [Cohere Transcribe](https://github.com/appautomaton/mlx-speech/blob/main/docs/cohere-asr.md) | Multilingual transcription | `cohere-asr` | [int8](https://huggingface.co/appautomaton/cohere-asr-mlx) |
| [Qwen3-ASR-1.7B](https://github.com/appautomaton/mlx-speech/blob/main/docs/qwen3-asr.md) | English, Chinese, and mixed speech | `qwen3-asr-1.7b` | [int8](https://huggingface.co/appautomaton/qwen3-asr-1.7b-int8-mlx) · [BF16](https://huggingface.co/appautomaton/qwen3-asr-1.7b-bf16-mlx) |
| [NVIDIA Nemotron 3.5 ASR Streaming](https://github.com/appautomaton/mlx-speech/blob/main/docs/nemotron-asr.md) | Multilingual streaming transcription | `nemotron-asr-streaming` | [int8](https://huggingface.co/appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx) |
| [IBM Granite Speech 4.0 1B](https://github.com/appautomaton/mlx-speech/blob/main/docs/granite-speech-asr.md) | Speech recognition with a selective-int8 language model | `granite-speech-4.0-1b` | [int8](https://huggingface.co/appautomaton/granite-4.0-1b-speech-int8-mlx) |

FireRedTTS3 Base bundles its speech model, codec, speaker encoder, and tokenizer
in `base/mlx-bf16/` within the FireRedTTS3 model repository. It produces mono
24 kHz audio and currently uses the GitHub source installation. See the
[FireRedTTS3 guide](https://github.com/appautomaton/mlx-speech/blob/main/docs/fireredtts3.md)
for conversion, generation, and measured runtime results.

<details>
<summary>Loading local weights, shared repositories, and DramaBox components</summary>

Flat model repositories accept an alias or a full repository ID.
`tts.load("fish-s2-pro")` and
`tts.load("appautomaton/fishaudio-s2-pro-8bit-mlx")` are equivalent. For a
repository containing multiple artifacts, use an alias or specify
`artifact_subdir`. Original checkpoint paths work where the model guide
documents their layout.

¹ DramaBox also downloads the
[Gemma 3 12B text encoder](https://huggingface.co/appautomaton/gemma-3-12b-it-backbone-4bit-mlx)
automatically. Its optional `denoise_ref=True` setting uses the MLX
[RE-USE / SEMamba enhancer](https://huggingface.co/appautomaton/re-use-semamba-mlx)
to clean noisy voice references. Denoising is off by default, and the enhancer
weights carry the NSCLv1 non-commercial license. The
[DramaBox guide](https://github.com/appautomaton/mlx-speech/blob/main/docs/dramabox.md)
covers these components and advanced controls.

</details>

## More examples

<details>
<summary>Python: voice cloning, streaming transcription, and model discovery</summary>

**Voice cloning with emotion tags**

```python
import mlx_speech
from mlx_speech.audio import write_wav

model = mlx_speech.tts.load("fish-s2-pro")
result = model.generate(
    "[excited] This is amazing!",
    reference_audio="reference.wav",
    reference_text="Transcript of the reference audio.",
)
write_wav("cloned.wav", result.waveform, sample_rate=result.sample_rate)
```

**Streaming transcription with Nemotron**

```python
import mlx_speech
from mlx_speech.audio import load_audio

nemotron = mlx_speech.asr.load("nemotron-asr-streaming")
session = nemotron.stream_session(language="en-US", att_context_size=(56, 3))
waveform, _ = load_audio("audio.wav", sample_rate=16_000, mono=True)
for start in range(0, int(waveform.size), 1_600):
    session.feed(waveform[start : start + 1_600])
session.finalize()
print(session.result().text)
```

**Granite transcription and model discovery**

```python
import mlx_speech

# Granite defaults to the published selective-int8 artifact
granite = mlx_speech.asr.load("granite-speech-4.0-1b")
print(granite.generate("audio.wav").text)

# Discover models
mlx_speech.tts.list_models()
mlx_speech.tts.list_models(detailed=True)  # includes shared-repo artifact paths
mlx_speech.asr.list_models()
```

</details>

<details>
<summary>CLI: waveform streaming, voice cloning, editing, and sound effects</summary>

```bash
# Bounded waveform streaming with dots.tts
mlx-speech tts --model dots-tts-soar --text "Hello!" --stream -o streamed.wav

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
# File transcription with the streaming-capable Nemotron model
mlx-speech asr --model nemotron-asr-streaming --audio speech.wav --language en-US
mlx-speech asr --model granite-speech-4.0-1b --audio speech.wav

# Local checkpoint paths work anywhere an alias does
mlx-speech tts --model models/fish_s2_pro/mlx-int8 --text "Hello!" -o output.wav
mlx-speech asr --model models/ibm/granite_4_0_1b_speech/mlx-int8 --audio speech.wav

# Discover models
mlx-speech tts --list-models
mlx-speech asr --list-models
mlx-speech --help
```

</details>

For sampling controls, diffusion steps, batch generation, and other advanced
options, follow the model's guide. Each guide names the supported controls and
the script that exposes them.

## Conversion

Use the published weights to get started. To convert an original checkpoint,
follow its model guide for source files, precision options, and the matching
[conversion script](https://github.com/appautomaton/mlx-speech/tree/main/scripts/convert).

Conversion runs separately from inference. Tools needed to read source
checkpoints are not runtime requirements.

## Development

```bash
git clone https://github.com/appautomaton/mlx-speech.git
cd mlx-speech
uv sync
uv run pytest
uv run ruff check .
```

The default test suite runs without model checkpoints. FireRedTTS3 includes
small golden fixtures for numerical regression checks, so those tests keep
working after large local weight files are removed. A manifest records the
checkpoint hashes and capture provenance. Real-weight inference and audio
quality use separate tests. See the
[testing guide](https://github.com/appautomaton/mlx-speech/blob/main/tests/README.md)
for test tiers and the
[FireRedTTS3 fixture guide](https://github.com/appautomaton/mlx-speech/blob/main/docs/fireredtts3.md#checkpoint-independent-regression-vectors)
for capture and regeneration.

```text
mlx-speech/
  src/mlx_speech/     library code
  scripts/           conversion, generation, eval, and audit entry points
  models/            local checkpoints (not in git)
  tests/             unit, checkpoint, runtime, integration tests
  docs/              model-family behavior guides
```

## License

Library code is released under the
[MIT license](https://github.com/appautomaton/mlx-speech/blob/main/LICENSE).
Model weights retain their respective licenses, listed in their model cards.

Built and maintained by [App Automaton](https://appautomaton.renocrypt.com).

## Acknowledgements

Thanks to the [linux.do](https://linux.do) community for its inspiration and
support.
