# FireRedTTS3 Base

FireRedTTS3 Base is a multilingual zero-shot voice-cloning model. The
`mlx-speech` implementation runs the complete path in MLX: CAM++ speaker
conditioning, RedAE waveform encoding, Qwen3/DiT latent generation, and RedAE
waveform reconstruction. Output is mono 24 kHz audio.

Only the Base checkpoint is supported. FireRedTTS3-Instruct voice design and
audio editing are outside this artifact and runtime.

## Model repository

The [FireRedTTS3 MLX repository](https://huggingface.co/appautomaton/fireredtts3-mlx)
groups models by variant and precision. The current Base bundle lives at
`base/mlx-bf16/` and includes all components required for inference. Instruct
can be added under its own directory when its MLX runtime is ready.

Install the current runtime from GitHub:

```bash
pip install "git+https://github.com/appautomaton/mlx-speech.git"
```

Load `fireredtts3-base` or the explicit `fireredtts3-base-bf16` alias. Both
select only `base/mlx-bf16/`. A full repository ID requires the subdirectory:

```python
from mlx_speech import tts

model = tts.load(
    "appautomaton/fireredtts3-mlx",
    artifact_subdir="base/mlx-bf16",
)
```

The loader downloads the selected bundle and the root model card. Future
variants in the repository do not become part of a Base download.

## Artifact layout

The runtime loads one flat directory:

```text
base/mlx-bf16/
  config.json
  core.safetensors
  redae.safetensors
  speaker.safetensors
  tokenizer.json
  tokenizer_config.json
  vocab.json
```

The three component weight files remain separate to avoid one oversized file;
they are not separate model repositories. Floating trainable weights use BF16.
RedAE's ISTFT window and CAM++ running statistics remain FP32.

Convert the pinned official files already staged under `models/`:

```bash
.venv/bin/python scripts/convert/fireredtts3.py \
  --input-dir models/firered/firered_tts3/original \
  --output-dir models/firered/firered_tts3/mlx-bf16
```

The converter reads the CAM++ PyTorch zip format with a restricted local
decoder. It does not import or require PyTorch.

## Voice cloning

The reference transcript must exactly describe the reference recording. Use a
reference in the same language as the target when possible.

```python
from mlx_speech import tts
from mlx_speech.audio import write_wav

model = tts.load("fireredtts3-base")
result = model.generate(
    "你好，很高兴认识你。",
    reference_audio="reference.wav",
    reference_text="For Timothy was a spoiled cat, and he allowed no one.",
    language="Chinese",
    seed=1234,
    guidance_scale=2.0,
    flow_steps=10,
    stop_threshold=0.5,
    max_audio_patches=400,
)
write_wav("generated.wav", result.waveform, sample_rate=result.sample_rate)
```

The same request is available through the unified CLI:

```bash
mlx-speech tts \
  --model fireredtts3-base \
  --text "你好，很高兴认识你。" \
  --reference-audio reference.wav \
  --reference-text "For Timothy was a spoiled cat, and he allowed no one." \
  --language Chinese \
  --seed 1234 \
  --guidance-scale 2.0 \
  --flow-steps 10 \
  --stop-threshold 0.5 \
  --max-audio-patches 400 \
  --output generated.wav
```

`max_new_tokens` is accepted as an alias for `max_audio_patches`. An in-memory
waveform defaults to 24 kHz; pass `reference_sample_rate` when it uses another
rate. File inputs carry their own sample rate and are resampled internally.

Local conversions still load directly from
`models/firered/firered_tts3/mlx-bf16`. To select the published bundle in the CLI
by repository ID, use `--model appautomaton/fireredtts3-mlx --artifact-subdir
base/mlx-bf16`.

## Inference contract

Each `generate()` call accepts one prepared utterance and returns one waveform.
The runtime performs the model-specific tokenization, reference conditioning,
autoregressive latent generation, and waveform decoding. It does not normalize
written text, detect its language, split long input, retry failed segments, or
join multiple waveforms. Those policies belong to the application that calls
the MLX runtime.

One generated audio patch contains four 64-dimensional RedAE frames. At 24 kHz,
each frame covers 480 waveform samples, so one patch represents 80 ms of audio.
The default 400-patch budget is therefore an approximate 32-second generation
ceiling, not a recommended utterance duration. For the local Mandarin fixture,
keeping prepared utterances below about 20 seconds, or roughly 40-60 Chinese
characters at its measured speaking rate, is a conservative application-level
operating rule rather than a universal model limit.

The repository's parity-oriented long smoke script demonstrates optional
caller-owned orchestration. It applies the upstream soft split targets
(`token_max_n=80`, `token_min_n=60`, `merge_len=20`), calls `generate()` once
per segment, and joins the resulting waveforms with a 50 ms linear cross-fade.
The library runtime does none of those operations implicitly, and the script's
upstream-compatible soft limits are not a recommended production scheduler.

## MLX runtime behavior

RedAE uses its official 64-token encoder and decoder sliding windows through a
window-bounded MLX attention path; the three-token CLS downsampler keeps full
attention. The Base core performs one Qwen prompt prefill followed by cached
one-token continuation. DiT uses BF16 linear compute and compiles only its
fixed 12-frame tensor region. Variable KV-cache tensors, stop-score host
synchronization, and request cleanup remain outside compiled code.

On the local golden request shown above, after one excluded warmup, three runs
recorded a 2.0518-second median core time and a 6,193,075,587-byte MLX peak.
That is 22.5% faster than the parity-correct 2.6478-second baseline and 19.7%
faster than the pre-parity 2.556-second profile, with peak memory effectively
unchanged. These are local comparison numbers, not cross-machine performance
claims.

The same golden gate checks two bitwise-identical seeded waveforms, exact local
ASR text (`你好，很高兴认识你。`), and CAM++ reference/output cosine of
0.7374. A short end-to-end runtime fixture also checks that post-cleanup active
memory stays within a 64 MiB spread across three consecutive requests on one
loaded model.

A local 514-Chinese-character experiment used seven official-style segments.
The MLX run produced 150.74 seconds of audio in 110.13 seconds (RTF 0.731) with
a 5.961 GiB MLX peak. Its matching PyTorch MPS reference produced 139.86 seconds
in 113.73 seconds (RTF 0.813). Listening found comparably severe noise late in
both outputs, so this fixture does not identify an MLX-only long-form
degradation. The numbers characterize this machine, voice, text, and
orchestration policy; they are not general performance or quality claims.

## Current limits

- Base voice cloning only; no Instruct tasks.
- Batch size one and non-streaming generation.
- One prepared utterance per request; long-form orchestration is caller-owned.
- The runtime does not include the optional upstream `wetext`, fastText, or
  LLM normalization layers. Supply normalized numbers and abbreviations when
  exact spoken forms matter.
- BF16 weights occupy about 5.7 GiB on disk across the three component files;
  generation also needs activation and KV-cache memory.

## Checkpoint-independent regression vectors

The committed `tests/fixtures/fireredtts3/` pack contains bounded golden
input/output vectors for tiny RedAE, CAM++, DiT, and two-patch autoregressive
core paths. The tests construct deterministic tiny-model weights from a fixed
formula, so no weight file is committed. They run from an empty working
directory and do not require `models/`, `.references/`, PyTorch, network access,
or a generated waveform.

Capture and replay use an explicit MLX CPU stream with the same strict
tolerances on developer Macs and CI. This avoids treating CPU/Metal numerical
differences as regressions. The manifest records the capture device. GPU
inference and waveform quality remain covered by the model's other tests.

The fixture manifest records the production BF16 artifact hashes and capture
provenance. These hashes identify the checkpoint used when the compatible MLX
behavior was accepted; deterministic micro-model parameters generated during
the test, rather than the 5.7 GiB artifact, drive routine regression coverage.

Capture the pack once while the production checkpoint is present:

```bash
.venv/bin/python scripts/audit/fireredtts3_golden_vectors.py capture \
  --model-dir models/firered/firered_tts3/mlx-bf16 \
  --output-dir tests/fixtures/fireredtts3
```

After implementation changes, regenerate into a temporary directory and compare
against the committed vectors:

```bash
.venv/bin/python scripts/audit/fireredtts3_golden_vectors.py regenerate \
  --model-dir models/firered/firered_tts3/mlx-bf16 \
  --compare tests/fixtures/fireredtts3
```

Run the checkpoint-independent gate directly:

```bash
pytest tests/unit/test_fireredtts3_golden_vectors.py
```

## PyTorch MPS parity reference

PyTorch is used only in the developer reference environment, never by the MLX
runtime or converter:

```bash
DYLD_LIBRARY_PATH=/opt/homebrew/opt/ffmpeg/lib \
PYTHONPATH=.references/FireRedTTS3 \
.venv-torch/bin/python scripts/audit/fireredtts3_reference.py \
  --model-dir models/firered/firered_tts3/original \
  --reference-audio /tmp/fireredtts3-mlx-port/reference.wav \
  --reference-text "For Timothy was a spoiled cat, and he allowed no one." \
  --text "你好，很高兴认识你。" \
  --language Chinese \
  --output /tmp/fireredtts3-mlx-port/reference-mps.wav
```
