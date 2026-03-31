# mlx-voice

Local speech synthesis on Apple Silicon, running pure MLX.

Four model families right now:

| Model | Best for |
| --- | --- |
| MossTTSLocal | shorter TTS, voice cloning, continuation |
| MOSS-TTSD | multi-speaker dialogue |
| MOSS-SoundEffect | text-to-sound-effect |
| VibeVoice | long-form speech, voice-conditioned generation |

Checkpoints live locally under `models/`. Scripts are the main interface.

## Setup

```bash
uv sync
```

Convert the checkpoints you want to use — each model family has a `scripts/convert_*.py` entry point.

## Usage

Generate with MossTTSLocal:

```bash
python scripts/generate_moss_local.py \
  --text "Hello, this is a test." \
  --output outputs/moss_local.wav
```

Clone a voice:

```bash
python scripts/generate_moss_local.py \
  --mode clone \
  --text "Hello, this is a cloned sample." \
  --reference-audio reference.wav \
  --output outputs/moss_local_clone.wav
```

Run a TTSD dialogue:

```bash
python scripts/generate_moss_ttsd.py \
  --text "[S1] Watson, we should go now." \
  --output outputs/ttsd.wav
```

Generate a sound effect:

```bash
python scripts/generate_moss_sound_effect.py \
  --ambient-sound "rolling thunder with steady rainfall on a metal roof" \
  --duration-seconds 8 \
  --output outputs/thunder.wav
```

Run VibeVoice:

```bash
python scripts/generate_vibevoice.py \
  --text "Hello from VibeVoice." \
  --output outputs/vibevoice.wav
```

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
mlx-voice/
  src/mlx_voice/    library code
  scripts/          conversion and generation entry points
  models/           local checkpoints (not in git)
  tests/            unit and integration tests
  docs/             model-family behavior guides
```
