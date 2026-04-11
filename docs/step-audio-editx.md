# Step-Audio-EditX

## What It Is

`Step-Audio-EditX` is the audio-editing family in this repo.

It is currently best thought of as:

- zero-shot voice cloning from prompt audio
- instruction-driven audio editing
- a decoder-LM + dual-tokenizer + CosyVoice non-stream waveform path

Current local runtime target:

- `models/stepfun/step_audio_editx/mlx-int8` (unified CLI default)
- `models/stepfun/step_audio_editx/original` (script default)

## Main Python API

Core entry point:

- `StepAudioEditXModel.from_dir(...)`
- `StepAudioEditXModel.from_path(...)`

Public inference methods:

- `clone(prompt_audio, prompt_sample_rate, prompt_text, target_text)`
- `edit(prompt_audio, prompt_sample_rate, prompt_text, edit_type, edit_info=None, target_text=None)`

The wrapper takes waveform arrays plus sample rates. File I/O stays in the CLI.

Return type:

- `StepAudioEditXResult`
  - `waveform`
  - `sample_rate`
  - `generated_token_ids`
  - `stop_reached`
  - `stop_reason`
  - `mode`

## Quick Start

```python
import mlx_speech

model = mlx_speech.tts.load("step-audio")
result = model.generate(
    "New cloned speech.",
    reference_audio="reference.wav",
    reference_text="Transcript of the reference audio.",
)
```

```bash
# Clone
mlx-speech tts --model step-audio \
  --reference-audio reference.wav \
  --reference-text "Transcript of the reference audio." \
  --text "New cloned speech." \
  -o cloned.wav

# Edit (emotion)
mlx-speech tts --model step-audio \
  --reference-audio reference.wav \
  --reference-text "Transcript of the reference audio." \
  --edit-type emotion --edit-info happy \
  -o happy.wav
```

Local path (skips HF download):

```bash
mlx-speech tts \
  --model models/stepfun/step_audio_editx/mlx-int8 \
  --reference-audio reference.wav \
  --reference-text "Transcript." \
  --text "New speech." \
  -o output.wav
```

## Script CLI (Advanced)

For `--temperature`, `--seed`, `--flow-steps`, and original-weight path:

- `scripts/generate/step_audio_editx.py`

Modes:

- `clone`
- `edit`

Current key flags:

- `--model-dir`
- `--tokenizer-dir`
- `--prompt-audio`
- `--prompt-text`
- `--output`
- `--max-new-tokens`
- `--temperature`
- `--seed`
- `--flow-steps`
- `--prefer-mlx-int8`

Important script default:

- the script defaults to **original** weights
- pass `--prefer-mlx-int8` to use the quantized path via the script

## Clone and Edit Semantics

`clone`

- prompt audio is tokenized with `vq02` + `vq06`
- prompt text and prompt audio tokens are packed into the shipped clone template
- Step1 generates new audio tokens
- flow + HiFT decode the final waveform

`edit`

- prompt audio is tokenized the same way
- the wrapper builds the shipped audio-edit prompt form
- supported edit types currently exposed:
  - `emotion`
  - `style`
  - `speed`
  - `denoise`
  - `vad`
  - `paralinguistic`

Current edit-type notes:

- `edit_info` is required for `emotion`, `style`, and `speed`
- `target_text` is required for `paralinguistic`

## Runtime Reality

What is landed locally now:

- Step1 MLX runtime
- Step-Audio dual tokenizer family (`vq02` + `vq06`)
- CosyVoice prompt-mel frontend
- CAMPPlus speaker embedding
- non-stream flow conditioning
- non-stream flow model
- HiFT waveform decode
- public wrapper + CLI

What is currently exercised:

- focused unit tests across tokenizer, Step1, frontend, CAMPPlus, flow, and HiFT
- gated local public-API smoke path through clone waveform generation
- gated local non-stream integration path through the lower-level runtime

## Reliability and Caveats

Current reliable path:

- non-stream local waveform generation from prompt audio + text through the MLX runtime

Current caveats:

- public integration coverage is gated and manual, not part of the default autorun suite
- `mlx-int8` packaging works, but original remains the safer default until parity is more thoroughly proven
- this family does not expose streaming in the current public API
- this family currently documents clone/edit only; no broader server/runtime surface is promised

## Example Usage

Python (low-level API):

```python
import numpy as np
from mlx_speech.audio.io import load_audio
from mlx_speech.generation import StepAudioEditXModel

audio, sr = load_audio("reference.wav", mono=True)
model = StepAudioEditXModel.from_dir(
    "models/stepfun/step_audio_editx/mlx-int8",
)
result = model.clone(
    np.asarray(audio, dtype=np.float32),
    sr,
    "Reference transcript.",
    "New cloned speech.",
)
```

Script CLI clone (advanced flags):

```bash
python scripts/generate/step_audio_editx.py \
  --prompt-audio reference.wav \
  --prompt-text "Reference transcript." \
  --output outputs/step_audio_clone.wav \
  clone \
  --target-text "New cloned speech."
```

Script CLI edit:

```bash
python scripts/generate/step_audio_editx.py \
  --prompt-audio noisy.wav \
  --prompt-text "Reference transcript." \
  --output outputs/step_audio_denoise.wav \
  edit \
  --edit-type denoise
```

## Local Validation

Manual public smoke:

```bash
RUN_LOCAL_INTEGRATION=1 uv run pytest tests/integration/test_step_audio_clone_public_api.py
```

This stays gated on purpose. It is useful for local confirmation, but it is not
part of the default fast test loop.
