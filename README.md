# mlx-speech

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/documentation/apple-silicon)

Local speech synthesis on Apple Silicon, running pure MLX. No cloud, no PyTorch.

| Model | Best for |
| --- | --- |
| MossTTSLocal | shorter TTS, voice cloning, continuation |
| MOSS-TTSD | multi-speaker dialogue |
| MOSS-SoundEffect | text-to-sound-effect |
| VibeVoice | long-form speech, voice-conditioned generation |

## Requirements

- Apple Silicon Mac (M1 or later)
- Python 3.13+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone https://github.com/appautomaton/mlx-speech.git
cd mlx-speech
uv sync
```

> PyPI package (`pip install mlx-speech`) coming soon.

Convert the checkpoints you want to use — each model family has a `scripts/convert_*.py` entry point:

```bash
python scripts/convert_moss_local.py
python scripts/convert_moss_audio_tokenizer.py
python scripts/convert_moss_ttsd.py
python scripts/convert_moss_sound_effect.py
python scripts/convert_vibevoice.py
```

## Usage

**Generate speech:**

```bash
python scripts/generate_moss_local.py \
  --text "Hello, this is a test." \
  --output outputs/moss_local.wav
```

**Clone a voice:**

```bash
python scripts/generate_moss_local.py \
  --mode clone \
  --text "Hello, this is a cloned sample." \
  --reference-audio reference.wav \
  --output outputs/moss_local_clone.wav
```

**Multi-speaker dialogue:**

```bash
python scripts/generate_moss_ttsd.py \
  --text "[S1] Watson, we should go now." \
  --output outputs/ttsd.wav
```

**Sound effect:**

```bash
python scripts/generate_moss_sound_effect.py \
  --ambient-sound "rolling thunder with steady rainfall on a metal roof" \
  --duration-seconds 8 \
  --output outputs/thunder.wav
```

**VibeVoice:**

```bash
python scripts/generate_vibevoice.py \
  --text "Hello from VibeVoice." \
  --output outputs/vibevoice.wav
```

## Exploring the Codebase

The PyPI package is still in progress. The best way to explore right now is to drop the repo into an agentic coding tool like [Claude Code](https://claude.ai/code) or Cursor — the codebase is structured and self-describing, and an agent can walk you through it quickly.

## Model Guides

Each family has a doc covering behavior, flags, and known limitations:

- [MossTTSLocal](./docs/moss-local.md)
- [MOSS-TTSD](./docs/moss-ttsd.md)
- [MOSS-SoundEffect](./docs/moss-sound-effect.md)
- [VibeVoice](./docs/vibevoice.md)

## Development

```bash
uv run pytest
uv run ruff check .
uv build --no-sources
```

```text
mlx-speech/
  src/mlx_voice/    library code
  scripts/          conversion and generation entry points
  models/           local checkpoints (not in git)
  tests/            unit and integration tests
  docs/             model-family behavior guides
```

## License

MIT — see [LICENSE](LICENSE)
