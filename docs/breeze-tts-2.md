# Breeze TTS 2 — local source and weight preparation

Status: the gitignored `mlx-bf16/` runtime package has been written.
Inference has not started. This is not a supported model adapter or a claim
of working MLX synthesis.

Implementation plan: [native MLX inference](../.agents/plans/breeze-tts-2-mlx.md).
Community model, codec, conversion, and cache snippets are staged under
`.references/{mlx-audio-breeze,mlx-breeze-tts2,BreezeTTS2_Mac_Streaming,breeze-tts-mlx}/`.
Each includes a `SOURCE.md` and upstream license. These are reading references;
none was installed or executed. Known hybrid PyTorch paths are excluded.

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
`scripts/convert/breeze_tts.py` writes `models/breezeblue/breeze_tts_2/mlx-bf16/`.
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
  No Breeze inference or parity run has occurred.

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
