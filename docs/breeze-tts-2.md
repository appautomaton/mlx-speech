# Breeze TTS 2

Status: local MLX inference is in the shared TTS API. Listening accepted
voice design, instruction direction, Peggy Chinese cloning, English and
Chinese Peggy direction, and streamed Chinese, including a 32.8 s Peggy line
at seed 3, heard again after the codec adopted its official 72-frame
attention window. English Peggy cloning and English `(sigh)` and `(laugh)`
events were accepted, in both a clone and a voice-design take. Warmed
streaming does not keep up with playback. Generated audio and probes belong
in `/tmp/breeze-tts-2-mlx/`, not in this repository.

Implementation plan: [native MLX inference](../.agents/plans/breeze-tts-2-mlx.md).
Community model, codec, conversion, and cache snippets are staged under
`.references/{mlx-audio-breeze,mlx-breeze-tts2,BreezeTTS2_Mac_Streaming,breeze-tts-mlx}/`.
Each includes a `SOURCE.md` and upstream license. These are reading references;
none was installed or executed. Known hybrid PyTorch paths are excluded.
[Sources](#sources) lists their public repositories.

## Inference

Load the gitignored `mlx-bf16/` or `mlx-int8/` package through the shared TTS
API. The runtime family is `breeze_tts`. The codec is `audio_tokenizer`, not
the Mimi block in `config.json`. Main weights stay BF16 and the codec stays
FP32; `mlx-int8/` packs the backbone and depth decoder projections as affine
int8 and is the faster choice.
`guidance_scale` and `cfg_scale` are the same control. Scale `1` keeps the
conditional branch. Any other scale requires `instruction`. A reference clip
requires its transcript. `max_new_tokens` is the codec-frame budget; the
default is 80, and the codec runs at 12.5 frames per second. Reaching the
budget before EOS raises a `RuntimeWarning` because the speech may be cut
off. `temperature` and `repetition_penalty` must be positive; `top_k` of 0
disables top-k. Codec attention uses the official 72-frame sliding window,
and a stream keeps only that window.

```python
import mlx_speech
from mlx_speech.audio import write_wav

model = mlx_speech.tts.load("models/breezeblue/breeze_tts_2/mlx-bf16")
reference_text = open(
    "outputs/source/peggy_hill_ref.txt", encoding="utf-8"
).read().strip()
clip = model.generate(
    "我们该把昨晚的事说清楚。",
    reference_audio="outputs/source/peggy_hill_ref.wav",
    reference_text=reference_text,
    instruction="说慢一点，语气克制、严肃。",
    guidance_scale=4.0,
    seed=42,
)
write_wav("peggy.wav", clip.waveform, sample_rate=clip.sample_rate)

for chunk in model.generate_stream(
    "我们该把昨晚的事说清楚。",
    reference_audio="outputs/source/peggy_hill_ref.wav",
    reference_text=reference_text,
    instruction="说慢一点，语气克制、严肃。",
    guidance_scale=4.0,
    seed=42,
    stream_chunk_patches=4,
):
    # chunk.waveform is 24 kHz mono PCM for the new frames only.
    ...
```

The CLI passes the reference transcript as `--reference-text`.

```bash
USE_TORCH=0 .venv/bin/python -m mlx_speech.tts.generate \
  --model models/breezeblue/breeze_tts_2/mlx-bf16 \
  --text "我们该把昨晚的事说清楚。" \
  --reference-audio outputs/source/peggy_hill_ref.wav \
  --reference-text "$(cat outputs/source/peggy_hill_ref.txt)" \
  --instruction "说慢一点，语气克制、严肃。" \
  --guidance-scale 4 \
  --seed 42 \
  --stream \
  --output /tmp/breeze-tts-2-mlx/wav/peggy.wav
```

## Batch generation

`generate_batch` generates independent utterances together and returns their
waveforms in input order. The default targets 256 compute branches: 256 requests
at `guidance_scale=1`, or 128 requests at any other scale. `batch_size` counts
requests and can override that target. Longer input lists are partitioned.

```python
import mlx.core as mx

mx.set_cache_limit(4 * 1024**3)
model = mlx_speech.tts.load("models/breezeblue/breeze_tts_2/mlx-int8")
clips = model.generate_batch(
    lines,
    reference_audio="outputs/source/peggy_hill_ref.wav",
    reference_text=reference_text,
    instruction="自然、清晰地朗读，语气温和。",
    guidance_scale=4.0,
    max_new_tokens=120,
    seed=42,
)
```

The voice reference, instruction and sampling controls are shared. Each request
has its own random key, repetition history and EOS; finished requests leave the
decode batch. Seeds default to `seed + input_index`; pass `seeds=[...]` to set
them explicitly. The reference is encoded once per call. Text encoding and
prefill handle unequal prompt lengths with masks and per-row positions. The
codec groups variable-length outputs into bounded batches and trims causal
padding before returning 24 kHz mono PCM.

MLX active allocation is only part of process memory. Its allocator can retain
freed buffers, especially as EOS compaction creates different KV-cache shapes.
Configure the process-wide pool explicitly as above when hosting batches; the
library leaves this shared allocator policy to the caller. On this M5 Max the
default free-cache limit was 121.6 GiB. Retaining different buffer sizes under
that default caused excessive process memory despite much smaller active peaks.

The benchmark defaults to that 4 GiB free-cache limit, clears unused warmup
buffers, and aborts if macOS reports a physical footprint above 48 GiB.
`--cache-limit-gib` and `--memory-guard-gib` control those values. Its JSON reports
active allocation, retained cache, current process footprint and lifetime peak
footprint separately. The process-footprint figures below include warmup; MLX
active peaks alone exclude retained cache and are not process-memory bounds.

One cache-size comparison at CFG 1, batch 256 and temperature 0.9 measured
29.01 s (25.59× real time, 21.38 GiB process peak) with a 4 GiB cache, versus
32.30 s (22.98×, 24.57 GiB) with an 8 GiB cache. At batch 512, an 8 GiB cache
measured 60.83 s (24.66×, 36.52 GiB), versus 62.60 s (23.97×, 32.18 GiB) at
4 GiB: about 2.9% higher throughput and 4.34 GiB more peak footprint in that
single comparison. Frame counts matched request by request, including the
capped take. These runs do not establish one universally optimal cache size;
4 GiB remains the benchmark default. The larger-cache records are
`cache8-no-cfg-{256,512}.json`.

Batching uses explicit random keys and the actual top-k threshold. Equal seeds
do not promise equal takes across the single-request and batch APIs; they use
different random streams, and batched kernels can round differently.
`generate_stream` continues to stream one utterance; `generate_batch` returns
complete waveforms.

### Host integration

Use the loaded model's public `generate_batch` method from a source checkout
that includes it. Keep one model loaded in the worker, configure the MLX cache
once at process startup, and submit independent script lines together. There is
no requirement for a helper process, JSON protocol or HTTP service.

The call returns `list[TTSOutput]` in input order. Each item contains an MLX
`waveform` array and `sample_rate` (24,000). Convert with
`np.asarray(item.waveform, dtype=np.float32)` when the host uses NumPy, or pass
the array directly to `mlx_speech.audio.write_wav`.

| Host requirement | Current batch contract |
| --- | --- |
| One voice and delivery per render | Shared `reference_audio`, `reference_text`, `instruction` and `guidance_scale` |
| Explicit seed for each line | `seeds=[...]`; a shared round seed is `seeds=[seed] * len(lines)` |
| A ceiling for each line | One `max_new_tokens`/`max_frames` ceiling per call; group equal budgets into separate calls when needed |
| Per-line frame count | `item.waveform.size // 1920` at 24 kHz |
| Completion metadata | `TTSOutput` has no finish-reason field; reaching the common frame ceiling warns and can be detected from the frame count |
| Silence detection while speaking | The existing single-line `generate_stream` supports host-side monitoring; the offline batch API does not stop on silence |
| A queue that admits more work as lines finish | Not part of this API; inputs are supplied per call and partitioned by `batch_size` |

Retry selection, ASR checks, text preparation, trimming, loudness normalization,
joining and manifests remain host responsibilities. Batch generation returns
every result without rerolling it. Existing host model locks should surround
model ownership as before. Calls from multiple threads are not a supported
replacement for `generate_batch`.

Batch memory grows with the longest prompt and the frame ceiling. The measured
256-request profile below used short lines and a 120-frame ceiling; grouping
different ceilings avoids reserving a long line's budget for every short line.

For a host with separate ANE and MLX backends, adapt each backend to the host's
existing multi-line method. The ANE runtime can keep its resident queue and
per-line options; MLX can use this in-process batch call. Their execution engines
do not need a common server protocol. Production code that requires per-line
ceilings and real-time silence stopping must account for the differences above;
the offline batch call is not a drop-in replacement for that streaming path.

Reproduce a target-size end-to-end measurement, including reference encoding,
prefill, autoregressive decoding and codec output:

```bash
uv run python scripts/eval/benchmark_breeze_batch.py \
  --reference-audio outputs/source/peggy_hill_ref.wav \
  --reference-text outputs/source/peggy_hill_ref.txt \
  --output /tmp/breeze-tts-2-mlx/batching/no-guidance.json
```

For 128 guided requests, add `--guidance-scale 4`. `--serial` renders the entire
same workload with `batch_size=1` for a measured comparison. The JSON retains
per-request frame counts and truncation warnings; timing summaries alone do not
establish that every take reached EOS. Selected WAVs are saved next to the JSON.
Throughput measurements retain every result, including capped takes, under the
fixed workload and frame budget. The benchmark does not run ASR, retry takes or
discard outputs based on their content.

Measured on 2026-11-13 on this M5 Max with 128 GiB memory and the int8 package:
three Mandarin lines repeated across the input batch, Peggy reference, temperature
0.9, top-k 50, repetition penalty 1.1, seeds starting at zero, and a 120-frame
budget. The guided run used the instruction above. These runs use a 4 GiB
free-cache limit and a 48 GiB process-footprint guard. Loading and warmup are
excluded from time; reference preparation, prefill, decoding and PCM output are
included.

| CFG | Request batch | Wall time | Audio | Aggregate throughput | Peak process footprint | Natural EOS |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 29.01 s | 742.48 s | 25.59× real time | 21.38 GiB | 256/256 |
| 1 | 512 | 62.60 s | 1500.24 s | 23.97× real time | 32.18 GiB | 511/512 |
| 4 | 128 | 32.79 s | 479.44 s | 14.62× real time | 21.81 GiB | 128/128 |
| 4 | 256 | 60.10 s | 969.68 s | 16.13× real time | 33.52 GiB | 256/256 |

Aggregate throughput sums audio duration across requests. Keep the default 256
requests without guidance for this workload: 512 did not improve throughput and
one take hit its budget. With guidance, explicitly setting `batch_size=256`
improved throughput over 128 by about 10%, at a higher memory cost. These are
workload-specific measurements, not hard batch limits. The earlier serial
baseline (256 requests, 605.13 s, 1.23× real time) was measured once without a
cache limit and reused; it was not rerun.

The four selected batch-256 no-guidance WAVs were byte-identical before and after
the cache-policy change, preserving their earlier exact Qwen3-ASR transcriptions.
Larger-batch spot checks are retained separately. The model can still stall or
misread words on other inputs or seeds; frame-budget warnings remain enabled.

TTS and ASR validation ran in separate, sequential processes. Raw measurements
and selected WAVs are under `/tmp/breeze-tts-2-mlx/batching/`:
`bounded-no-cfg-{256,512}.json`, `bounded-cfg4-{128,256}.json`,
`bounded-quality.json`, and `final-no-cfg-256.json` (serial baseline). Earlier
probes marked invalid after the padded-prefill NaN diagnosis are excluded.

## Measured runtime

These numbers are from one warmed process on this Mac. Loading the package
took 0.75 s and is not included below. Each measured run streams in chunks
of four codec frames. MLX active memory stayed near 7.1 GiB. Encoding the
Peggy reference raised the MLX cache to about 1.8 GiB and the process
physical footprint to about 9.2 GiB.

| Run | First audio | Utterance | RTF |
| --- | --- | --- | --- |
| English design, no CFG, seed 0 | 0.41 s | 3.84 s, 48 frames, EOS | 1.12 |
| English direction, CFG 4, seed 0, no reference | 0.54 s | hit the 120-frame cap | 1.60 on that capped run |
| Chinese design, no CFG, seed 42 | 0.43 s | 1.44 s, 18 frames, EOS | 1.21 |
| Chinese direction, CFG 4, seed 42 | 0.55 s | 1.68 s, 21 frames, EOS | 1.62 |
| Peggy clone, no CFG, seed 42 | 0.51 s | 2.08 s, 26 frames, EOS | 1.21 |
| Peggy direction, CFG 4, seed 42 | 0.73 s | 2.00 s, 25 frames, EOS | 1.78 |

Those runs predate the speed work below. No-CFG real-time factor was about
1.1 to 1.2, and CFG scale 4 about 1.6 to 1.8, on these short lines.

### Batched guidance, compiled kernels and int8

Under guidance the conditional and unconditional branches run as one batch of
two, in the backbone and the depth decoder alike. The shorter prompt is
left-padded and masked, and each row keeps its own rotary positions. Norm,
rotary and SwiGLU kernels are compiled (`mx.compile`, about 0.35 s on first
use in a process). Without guidance, output is bit-identical to the eager
path. With guidance, batched kernels round differently in BF16, so a seed
samples a different, equally valid take.

Measured on 2026-11-06 in one warmed process: three narration lines, five
seeds each, a cloned reference, temperature 0.9. English used an instruction
at CFG 4; Chinese used none.

| Build | English, CFG 4 RTF | Chinese, no CFG RTF | Peak memory |
| --- | --- | --- | --- |
| Before, BF16 | 0.863 | 0.632 | 11.08 GiB |
| After, BF16 | 0.593 | 0.564 | 11.08 GiB |
| After, int8 | 0.475 | 0.444 | 9.01 GiB |

Audio8 ASR heard the same words from every build: no English errors in BF16
and 3 of 835 characters in int8 ("big estates" heard as "biggest states"),
and one Chinese take in five with 3 of 160 characters wrong in each build.

## Known limitations

- Generation can miss EOS and keep sampling silence until the frame budget.
  It is most common in English and under CFG. In the measurement above,
  English direction at seed 0 without a reference ran to the frame cap. An
  earlier English Peggy direction at seed 42 did the same; seed 0 with that
  reference finished. A plain English Peggy clone spoke its whole sentence,
  then ran to the cap at seeds 0 and 42. The 32.8 s Chinese Peggy line ran to
  its 700-frame cap at seeds 42 and 0; seed 3 reached EOS at 410 frames. The
  official inference stops only on backbone EOS or the length limit, as this
  port does.
- An English vocal event can swallow the words after it. A Peggy clone of
  `(sigh) It is good to hear your voice again after all this time.` gave
  about 0.5 s of sound and no words at seed 0. A closing
  `(clears throat) All right, listen up.` was dropped in all eight clone and
  design attempts. Takes with `(sigh)` and `(laugh)` before or between
  sentences did say every word; one clone and one design take reached EOS.
- Streamed codec audio matches one offline decode of the same codes except for
  overlap-add and attention rounding. The 32.8 s Peggy line is about 63 dB SNR
  against its offline render.
- There is no published hub alias yet. Point the loader at the local
  `mlx-bf16` directory.
- The BreezeBlue non-commercial license still applies to the weights and to
  outputs made from them.

## Implementation constraint

For this model, do not install, import, or execute PyTorch or torchaudio for
inference, conversion, parity tests, fixture generation, or validation.
Read the official code as source and implement inference directly in MLX for
Apple Silicon. Use safetensors metadata, MLX/NumPy numerical checks, and
eventual waveform and streaming validation; do not run a PyTorch oracle.
Do not install upstream `requirements.txt` or add its packages as runtime
dependencies. The downloaded source distributions are for reading only.

## Local assets and provenance

| Asset | Local path | Pinned version |
| --- | --- | --- |
| [Official inference source](https://github.com/breezeblue-ai/breeze-tts) | `.references/breeze-tts/` | `008f769016b0a24711becd7a4925030bc93f608c` |
| [Official weights](https://huggingface.co/BreezeBlue/Breeze-TTS-2) | `models/breezeblue/breeze_tts_2/original/` | `3e28c5151381a722f1d8661b4118c298caa77aa4` |
| [Qwen codec source release](https://pypi.org/project/qwen-tts/0.1.1/) | `.references/qwen-tts-0.1.1/` | `qwen-tts==0.1.1` |
| [Transformers source release](https://pypi.org/project/transformers/4.57.3/) | `.references/transformers-4.57.3/` | `transformers==4.57.3` |

The two source distributions match the versions in the official inference
repository's `requirements.txt`. Their archive SHA-256 values are:

- `qwen_tts-0.1.1.tar.gz`:
  `afba5fa235806a6883f46a389e67540b46f8a55da457216bf1d7342903814780`
- `transformers-4.57.3.tar.gz`:
  `df4945029aaddd7c09eec5cad851f30662f8bd1746721b34cc031d70c65afebc`

All source snapshots and weights are gitignored. The existing
`.references/transformers/` checkout is independent and remains unchanged.
Keep original weights intact and separate from the converted package.
`scripts/convert/breeze_tts.py` writes `models/breezeblue/breeze_tts_2/mlx-bf16/`;
with `--int8` it writes `mlx-int8/`, whose `config.json` names each of the 280
int8 projections. Embeddings, norms, output heads, the depth input projector
and the text encoder stay BF16 there.
Main tensors stay BF16. The bundled codec stays FP32, with convolution kernels
in MLX layout. Mimi `codec_model` weights are left out of that package.
Retain the nested `original/audio_tokenizer/` directory: the official loader
resolves the codec there. No additional codec-weight download is required.

## Reproduce the weight download

Run from the repository root using the standalone Hugging Face CLI. The
explicit CLI request for this preparation overrides the generic preference
to avoid `hf`; it does not add a CLI dependency to the library.

```bash
USE_TORCH=0 HF_HUB_DISABLE_XET=1 hf download BreezeBlue/Breeze-TTS-2 \
  --revision 3e28c5151381a722f1d8661b4118c298caa77aa4 \
  --local-dir models/breezeblue/breeze_tts_2/original \
  --max-workers 4

USE_TORCH=0 hf cache verify BreezeBlue/Breeze-TTS-2 \
  --revision 3e28c5151381a722f1d8661b4118c298caa77aa4 \
  --local-dir models/breezeblue/breeze_tts_2/original \
  --fail-on-missing-files
```

The CLI writes `.cache/huggingface/` metadata inside `--local-dir`. With
`hf` 1.7.2, `--fail-on-extra-files` flags that metadata as extra files, so
omit that flag and check the source-file inventory excluding `.cache/`.
Keep the metadata for resumable downloads; it is not part of the checkpoint.

Download the entire 17-file snapshot, including configuration, tokenizer,
shard index, license, and bundled codec. Main safetensors are BF16 with a
small FP32 buffer; the separate audio tokenizer safetensors are FP32. These
are original floating-point weights, not INT8 or 4-bit artifacts.

Preparation verification:

- Hugging Face CLI 1.7.2 verified all 17 remote files: every checksum matches.
  Excluding CLI metadata, local paths and byte sizes match the pinned API
  inventory exactly: 7,683,635,994 bytes.
- Main shard index resolves all 1,115 tensor entries; the separate audio
  tokenizer contains 496 tensors. Header shapes, dtypes, data offsets, and
  file sizes were checked without importing a model framework.
- Main tensors: 3,483,206,465 BF16 elements and 32 FP32 elements; audio
  tokenizer: 170,557,441 FP32 elements. These are stored tensor-element counts,
  including buffers, not a claim about trainable or active runtime parameters.
- Official checkout matches the pinned commit and has no local changes.
  Both dependency source archives matched their PyPI SHA-256 values.
- Existing project baseline, before this converter: `USE_TORCH=0 .venv/bin/pytest tests/unit/`
  passed 1,184 tests. The converter has since written `mlx-bf16/`.
  Inference landed after that preparation check.

## Source entry points

- `breeze-tts/infer.py` and `breeze_infer/runtime.py`: CLI and component loading.
- `breeze-tts/breeze_infer/templates.py` and `audio.py`: instruction/reference
  conditioning and reference-audio preparation.
- `breeze-tts/models/t5gemma2_compat.py`: text encoder implementation.
- `breeze-tts/models/breeze_backbone_factory.py` and `breeze.py`: backbone
  selection, audio embeddings, depth decoder, and model assembly.
- `breeze-tts/models/fast_streaming.py` and `models/stream_runtime/`: streaming
  generation, cache/state handling, and codec output scheduling.
- `qwen-tts-0.1.1/qwen_tts/inference/qwen3_tts_tokenizer.py` and
  `qwen_tts/core/tokenizer_12hz/`: bundled audio tokenizer implementation.
- `transformers-4.57.3/src/transformers/models/qwen3/` and `models/mimi/`:
  external model definitions referenced by the official graph.

The paths above are relative to `.references/`. Inspect actual loader and
generation usage before deciding which checkpoint tensors the MLX path needs;
the main config also contains a Mimi codec alongside the separately loaded
Qwen audio tokenizer. Do not infer the active codec from config names alone.

The target capabilities are English/Chinese voice cloning (reference audio
plus exact transcript), reference-free voice design, voice direction,
inline vocal events, and 24 kHz waveform/streaming output. Availability in
upstream is not evidence these capabilities already work in `mlx-speech`.

## Licenses

Official source is Apache-2.0. Breeze model weights, derivative models, and
self-hosted outputs use the bundled BreezeBlue Research and Non-Commercial
License. The source-only dependencies retain their own license files.
MLX conversion does not remove the model's non-commercial terms.

## Sources

- Official inference: [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts)
  at `008f769`, with `qwen-tts==0.1.1` and `transformers==4.57.3` for the codec
  and backbone definitions. It is the behavior reference for prompts, CFG,
  sampling, EOS, and the codec, including its 72-frame attention window.
- Official weights: [BreezeBlue/Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2)
  at `3e28c51`.

Earlier community MLX work was read for implementation ideas at these
revisions. None was installed or run, and none is a runtime dependency.

- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) `4ab7e6f`: Breeze
  graph and Qwen codec in MLX. Its depth loop recomputes the frame prefix and
  reads each sampled code back to the host.
- [vanch007/mlx-breeze-tts2](https://github.com/vanch007/mlx-breeze-tts2)
  `51c182d`: standalone model, codec, weight mapping, cached depth decoding,
  and stream cleanup. It tokenizes through Transformers.
- [xzf-thu/BreezeTTS2_Mac_Streaming](https://github.com/xzf-thu/BreezeTTS2_Mac_Streaming)
  `c6552ad`: depth KV reuse and device-side sampling, layered on mlx-audio.
- [rishikksh20/breeze-tts-mlx](https://github.com/rishikksh20/breeze-tts-mlx)
  `4e045c1`: MLX text encoder, backbone, and depth decoder. Its codec runs in
  PyTorch.

This port keeps the whole path in MLX: backbone KV across frames, a fresh
depth KV per frame, on-device sampling, and a windowed streaming codec. It
has not been benchmarked against these projects.
