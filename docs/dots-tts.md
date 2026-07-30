# dots.tts

`dots.tts` is a continuous autoregressive TTS family with a Qwen2.5 text
backbone, continuous latent patches, SOAR or MeanFlow acoustic solving, CAM++
speaker conditioning, and a 48 kHz AudioVAE waveform decoder. The
`mlx-speech` implementation performs inference entirely in MLX and does not
import PyTorch, Transformers, `mlx-lm`, or either reference checkout.

## Checkpoint selection

All four runtime artifacts live in `appautomaton/dots-tts-mlx`.

| Alias | Remote path | Stored precision | Gate size | Gate peak |
| --- | --- | --- | ---: | ---: |
| `dots-tts-soar` | `soar/mlx-int8/` | Qwen-selective int8 | 3.210 GiB | 6.963 GiB |
| `dots-tts-soar-base` | `soar/mlx-base/` | Mixed BF16/FP32 | 4.557 GiB | 8.308 GiB |
| `dots-tts-mf` | `mf/mlx-int8/` | Qwen-selective int8 | 3.212 GiB | 7.177 GiB |
| `dots-tts-mf-base` | `mf/mlx-base/` | Mixed BF16/FP32 | 4.559 GiB | 8.521 GiB |

The short aliases select int8 because both int8 artifacts passed the fixed
English/Mandarin cloning gate. Use an explicit `-base` alias when debugging
conversion or comparing against the source-faithful mixed-precision artifact.

SOAR uses a 10-step flow-matching solver by default and exposes classifier-free
guidance. MeanFlow is a four-step distilled solver without a separate runtime
guidance branch. The reproduced gate did not establish a universal quality or
speed winner; choose the solver family for the behavior you need, then compare
it on your own voices and text.

The size and peak figures above come from the
[2026-07-30 local quantization gate](benchmarks/dots-tts-quant-gate-2026-07-30.md).
They are observations from that workload, not minimum system-memory
requirements.

## Voice cloning

Continuation cloning uses reference audio and its matching transcript. It
encodes prompt latents and speaker identity before generating the target.

```python
from mlx_speech import tts
from mlx_speech.audio import write_wav

model = tts.load("dots-tts-soar")
result = model.generate(
    "Today the weather is bright and peaceful.",
    reference_audio="reference.wav",
    reference_text="My name is Samantha. I speak clearly and calmly.",
    language="en",
    max_audio_patches=128,
    seed=42,
)
write_wav("output.wav", result.waveform, sample_rate=result.sample_rate)
```

The general CLI supports this mode:

```bash
mlx-speech tts \
  --model dots-tts-soar \
  --text "Today the weather is bright and peaceful." \
  --reference-audio reference.wav \
  --reference-text "My name is Samantha. I speak clearly and calmly." \
  --max-new-tokens 128 \
  --output output.wav
```

Speaker-only cloning uses a reference waveform without a transcript. It uses
the CAM++ speaker embedding but does not prefill the reference speech latents.
The current general CLI requires audio and transcript together, so use the
Python API for speaker-only cloning:

```python
model = tts.load("dots-tts-mf")
result = model.generate(
    "今天的天气晴朗而平静。",
    reference_audio="reference.wav",
    language="zh",
)
```

Passing neither reference keeps the official target-only schedule, but the
resulting random voice was not included as a quality-supported mode in the
release gate.

Audio paths are loaded as mono and resampled internally to 48 kHz. An in-memory
MLX waveform also works; pass `reference_sample_rate` when its rate is not
48 kHz. The bounded sinc resampler accepts positive integer input rates and
does not require the source rate to divide 48 kHz evenly.

## Generation controls

These dots.tts controls are keyword arguments to `model.generate(...)`.

| Argument | Default | Behavior |
| --- | ---: | --- |
| `max_audio_patches` | 128 | Hard cap on generated latent patches; `max_new_tokens` is an alias. |
| `solver_steps` | SOAR 10, MF 4 | Positive override for acoustic solver evaluations. |
| `guidance_scale` | 1.2 | SOAR classifier-free guidance; accepted but unused by MeanFlow. |
| `speaker_scale` | 1.5 | Scales the reference speaker embedding. |
| `language` | `None` | Optional two- or three-letter language tag such as `en`, `zh`, or `yue`. |
| `seed` | 42 | Seeds latent sampling for repeatable runs on the same stack. |
| `eos_threshold` | 0.8 | Qwen EOS probability threshold in the closed interval `[0, 1]`. |
| `template` | `tts` | `tts` or the source-compatible `tts_interleave` schedule. |

Continuation references consume part of the patch budget. The runtime rejects
a budget that cannot hold the reference prefill, its regenerated tail, and at
least one target patch. Raising the budget can increase both output length and
peak memory.

## Artifact layout and precision

The local and remote runtime layouts are identical:

```text
models/dots_tts/
  soar/
    mlx-base/
    mlx-int8/
  mf/
    mlx-base/
    mlx-int8/
```

Each artifact is self-contained:

```text
<variant>/<artifact>/
  config.json
  llm_config.json
  mlx_config.json
  core.safetensors
  vocoder.safetensors
  speaker.safetensors
  latent_stats.safetensors
  tokenizer/
```

`mlx-base` is intentionally mixed precision. Core modules and the AudioVAE
decoder are BF16. CAM++, latent statistics, the AudioVAE encoder,
`enc_mi_layer`, and `pre_proj` remain FP32. These FP32 paths were retained
because converted-weight parity testing did not support reducing them.

`mlx-int8` applies affine 8-bit quantization with group size 64 only to eligible
native `qwen.model.*` Linear and Embedding modules. It stores packed weights as
U32 and scales/biases as BF16. EOS, semantic and acoustic modules, conditioning
projections, CAM++, AudioVAE, and latent statistics keep their exact
`mlx-base` dtypes. It is not a whole-model 8-bit artifact.

## Reproducing conversion

Weights stay under the ignored `models/` directory. Download and verify the
pinned official checkpoints, build the base artifacts, then derive int8 from
the verified base:

```bash
uv run python scripts/convert/download_dots_tts.py --variant all --verify
uv run python scripts/convert/dots_tts.py --variant all --precision base
uv run python scripts/audit/dots_tts_checkpoint.py --variant all --precision base
uv run python scripts/convert/dots_tts.py --variant all --precision int8
uv run python scripts/audit/dots_tts_checkpoint.py --variant all --precision int8
```

Conversion is transactional and refuses to overwrite a non-empty artifact
directory. Remove or relocate an existing target intentionally before
rebuilding it. The converter uses the official checkpoints as input, but the
published runtime and converted checkpoint loader remain Torch-free.

## Reproduced quality gate

The fixed corpus contains one macOS English voice and one macOS Mandarin voice,
each evaluated in continuation and speaker-only modes. It is a small regression
gate, not a broad benchmark.

| Variant | Base WER | Int8 WER | Base speaker cosine | Int8 speaker cosine |
| --- | ---: | ---: | ---: | ---: |
| SOAR | 0.0000 | 0.0000 | 0.7992 | 0.8147 |
| MeanFlow | 0.0588 | 0.0588 | 0.7868 | 0.7901 |
| Overall | 0.0294 | 0.0294 | 0.7930 | 0.8024 |

The release thresholds were absolute WER regression ≤ 0.01 and speaker-cosine
regression ≤ 0.02. Both int8 artifacts passed. These numbers are locally
reproduced measurements; they do not claim equality for other prompts,
speakers, languages, seeds, or hardware.

## Memory and limitations

- Generation is non-streaming and continuously autoregressive. Prompt and
  generated history remain live, so memory grows with reference duration and
  patch budget even when Qwen is int8.
- Load and evaluate one artifact at a time. The four-artifact gate did this
  sequentially and stopped runs above explicit memory guards.
- Start with the int8 alias and a bounded patch budget. The measured int8 peaks
  were about 7.0–7.2 GiB for the 128-patch gate; longer or different workloads
  may use more.
- Output is finite, mono, 48 kHz audio. There is no public streaming API and no
  training path in `mlx-speech`.
- English and Mandarin passed the local release gate. The tokenizer can accept
  other language tags, but this release does not publish MLX quality results
  for them.
- Reference noise, clipping, inaccurate transcripts, unusual prosody, and very
  short recordings can reduce cloning quality.

## Responsible use

Only clone a voice when the speaker has authorized that use. Treat reference
recordings as biometric data and control their storage and access. Clearly
disclose synthetic speech to listeners. Do not use the model for impersonation,
fraud, deceptive attribution, harassment, or disinformation. Deployment owners
remain responsible for consent, applicable law, abuse monitoring, and any
additional safeguards their context requires.

## Sources

- Official implementation: `studio-dots-ai/dots.tts`, pinned locally at commit
  `5ed719e3d36f5a3f6d8037ca9a7009d4fd0520ba` (`v0.2.1`)
- Community MLX comparison: `sb1992/dots-tts-mlx`, pinned at
  `f64479f51a2a9d7093533732cae86e765d8fb96e` (`v0.7.0`)
- SOAR weights: revision
  `e3520f75254d0020a0406db31c51a79d00d22d55`
- MeanFlow weights: revision
  `25c53fb462e57087e52237daa5ea30df1c5cc328`

See [reference provenance](references.md) for the local checkout paths and
roles.
