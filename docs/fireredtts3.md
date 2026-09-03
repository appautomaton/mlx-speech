# FireRedTTS3 Base

FireRedTTS3 Base is a multilingual zero-shot voice-cloning model. The
`mlx-speech` implementation runs the complete path in MLX: CAM++ speaker
conditioning, RedAE waveform encoding, Qwen3/DiT latent generation, and RedAE
waveform reconstruction. Output is mono 24 kHz audio.

Only the Base checkpoint is supported. FireRedTTS3-Instruct voice design and
audio editing are outside this artifact and runtime.

## Local artifact

The runtime loads one flat directory:

```text
mlx-bf16/
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

model = tts.load("models/firered/firered_tts3/mlx-bf16")
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
  --model models/firered/firered_tts3/mlx-bf16 \
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

## Current limits

- Base voice cloning only; no Instruct tasks.
- Batch size one and non-streaming generation.
- The runtime expects already-normalized, single-utterance text and does not
  reproduce the optional upstream `wetext`/LLM normalization layer.
- BF16 weights occupy about 5.7 GiB on disk across the three component files;
  generation also needs activation and KV-cache memory.

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
