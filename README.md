# mlx-voice

`mlx-voice` is an MLX-native speech library for Apple Silicon.

The initial target is the OpenMOSS voice family, with room to add more speech
models over time behind a consistent MLX-first interface.

## Status

v0 is a working inference library. `MossTTSLocal` runs end-to-end from text
to 24 kHz waveform using pure MLX on Apple Silicon.

**What is implemented:**

- Pure-MLX runtime — no PyTorch dependency
- `MossTTSLocal` end-to-end waveform generation
- Cat audio tokenizer encode and decode
- Four inference modes: direct generation, clone, continuation, continuation + clone
- Processor-side reference audio encode/decode helpers
- Default local-path loading from `models/openmoss/.../mlx-int8/`
- Default global + local KV cache for single-item sampled inference
- `MossTTSDelay` / TTSD MLX inference path with default KV cache and
  quantized-first runtime selection

**Runtime characteristics:**

- `mlx-int8` quantized checkpoints by default
- `W8Abf16` mixed precision: quantized weights, `bfloat16` activations,
  `float32` preserved for numerically sensitive reductions (RMSNorm, attention
  scores, LFQ distances)
- KV cache on by default: `~2.32x` speedup over uncached, `~1.64x` faster
  than real time on quantized hardware
- Uncached path available via `--no-kv-cache` for debug and comparison

## Design Principles

- MLX-first, not torch-wrapped
- Minimal dependencies by default
- Local-path-first model loading
- Clean separation between runtime inference and checkpoint conversion
- Model adapters behind a stable library surface
- Apple Silicon as the primary target

## Dependency Philosophy

- `mlx`: yes
- `numpy`: yes
- `safetensors`: yes
- `torch`: no
- `torchaudio`: no
- `huggingface_hub`: avoid until the Python library itself truly needs it
- `mlx-audio`: reference project, not a required dependency

## Usage

### Prerequisites

Download and convert the upstream checkpoints once:

```bash
# Convert speech model
python scripts/convert_moss_local.py

# Convert audio tokenizer
python scripts/convert_moss_audio_tokenizer.py
```

Converted weights land in `models/openmoss/moss_tts_local/mlx-int8/` and
`models/openmoss/moss_audio_tokenizer/mlx-int8/`.

TTSD conversion:

```bash
python scripts/convert_moss_ttsd.py
```

Converted TTSD weights land in `models/openmoss/moss_ttsd/mlx-int8/`.

### Direct generation

```bash
python scripts/generate_moss_local.py \
  --text "Hello, this is a test." \
  --output output.wav
```

KV cache is on by default. To disable:

```bash
python scripts/generate_moss_local.py \
  --text "Hello, this is a test." \
  --no-kv-cache \
  --output output.wav
```

### Clone and continuation

```bash
# Clone a voice
python scripts/generate_moss_local.py \
  --mode clone \
  --text "Hello, this is a test." \
  --reference-audio reference.wav \
  --audio-temperature 1.0 \
  --audio-top-p 0.95 \
  --audio-top-k 50 \
  --audio-repetition-penalty 1.1 \
  --output output.wav

# Continue from existing audio
python scripts/generate_moss_local.py \
  --mode continuation \
  --text "Hello, this is a test." \
  --reference-audio reference.wav \
  --output output.wav
```

### TTSD generation

```bash
python scripts/generate_moss_ttsd.py \
  --text "[S1] Watson, we should go now." \
  --output output.wav
```

`generate_moss_ttsd.py` prefers local `mlx-int8` TTSD and codec artifacts by
default. To force the original reference weights instead:

```bash
python scripts/generate_moss_ttsd.py \
  --prefer-original \
  --text "[S1] Watson, we should go now." \
  --output output.wav
```

### Clone evaluation

Materialize the fixed English clone-eval reference set from macOS built-in
voices:

```bash
python scripts/materialize_clone_eval_macos.py \
  --manifest examples/clone_eval/macos_builtin_en.json \
  --output-dir outputs/clone_eval/macos_builtin_en
```

Sweep clone presets on that eval set:

```bash
python scripts/sweep_clone_presets.py \
  --manifest examples/clone_eval/macos_builtin_en.json \
  --reference-dir outputs/clone_eval/macos_builtin_en/references \
  --output-dir outputs/clone_eval/macos_builtin_en/runs
```

## Development

Install the project and development dependencies with `uv`:

```bash
uv sync
```

Useful commands:

```bash
uv run pytest
uv run ruff check .
uv build --no-sources
```

## Repository Layout

```text
mlx-voice/
  src/mlx_voice/      # Published library code
  models/             # Local checkpoints, not packaged
  plans/              # Versioned implementation plans
  tests/              # Focused package tests
  examples/           # Small usage examples
  scripts/            # Conversion and maintenance helpers
  docs/               # Internal project notes
  .references/        # Local read-only upstream checkouts, not packaged
```

## Non-Goals For V0

- Supporting every speech model at once
- Supporting `.bin` checkpoints
- Building a large umbrella audio framework
- Pulling in broad dependency stacks before they are needed
- Reintroducing PyTorch into the implementation path
