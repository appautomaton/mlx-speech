# Hugging Face Release Workflow

This document describes how `mlx-speech` runtime artifacts are published to
Hugging Face.

The repository stays code-only. Converted weights, tokenizers, and processor
assets are published through Hugging Face model repos under the
`appautomaton` org.

## Published Repos

Each model family gets its own Hugging Face repo.

| Local model directory | Hugging Face repo | Published variant |
| --- | --- | --- |
| `models/openmoss/moss_audio_tokenizer/` | `appautomaton/openmoss-audio-tokenizer-mlx` | `mlx-int8` |
| `models/openmoss/moss_tts_local/` | `appautomaton/openmoss-tts-local-mlx` | `mlx-int8` |
| `models/openmoss/moss_ttsd/` | `appautomaton/openmoss-ttsd-mlx` | `mlx-int8` |
| `models/openmoss/moss_sound_effect/` | `appautomaton/openmoss-sound-effect-mlx` | `mlx-4bit` |
| `models/vibevoice/` | `appautomaton/vibevoice-mlx` | `mlx-int8` |
| `models/cohere/cohere_transcribe/` | `appautomaton/cohere-asr-mlx` | `mlx-int8` |
| `models/Qwen3-ASR-1.7B-MLX-BF16/` | `appautomaton/qwen3-asr-1.7b-bf16-mlx` | bf16, repo root |
| `models/ibm/granite_4_0_1b_speech/mlx-int8/` | `appautomaton/granite-4.0-1b-speech-int8-mlx` | selective int8, repo root |
| `models/dots_tts/` | `appautomaton/dots-tts-mlx` | SOAR/MF × `mlx-base`/`mlx-int8` |

Multi-variant family repositories keep variants in explicit subfolders.
Single-variant ASR repositories such as Qwen3-ASR and Granite publish one
self-contained runtime package flat at the repo root.

## Release Boundaries

- Do not commit large checkpoint files into this repository.
- Keep each Hugging Face repo focused on one runtime artifact family.
- Keep original upstream checkpoints out of the published MLX repo layout.
- Only publish artifacts that have already been validated locally.

## Expected Repo Layout

Typical Hugging Face layout:

```text
appautomaton/<model-name>-mlx/
  README.md
  mlx-int8/
    config.json
    model.safetensors
    ...
```

Sound effect currently uses:

```text
appautomaton/openmoss-sound-effect-mlx/
  README.md
  mlx-4bit/
    config.json
    model.safetensors
    ...
```

Qwen3-ASR BF16 and Granite selective int8 each publish one package at the repo
root. Granite uses this layout:

```text
appautomaton/granite-4.0-1b-speech-int8-mlx/
  README.md
  config.json
  model.safetensors
  preprocessor_config.json
  processor_config.json
  tokenizer_config.json
  tokenizer.json
  vocab.json
  merges.txt
```

dots.tts uses one family repository with four self-contained artifacts:

```text
appautomaton/dots-tts-mlx/
  README.md
  soar/
    mlx-base/
    mlx-int8/
  mf/
    mlx-base/
    mlx-int8/
```

The dots.tts upload target constructs only these four explicit include
patterns. It cannot select local `original/` checkpoints or obsolete
`mlx-bf16/` directories. Before publishing, validate hashes, required runtime
files, the model card, and the checked benchmark verdict without network access:

```bash
uv run python scripts/hugging_face/upload.py dots-tts --dry-run
```

The dry-run manifest must list exactly `soar/mlx-base`, `soar/mlx-int8`,
`mf/mlx-base`, and `mf/mlx-int8` under `appautomaton/dots-tts-mlx`. It fails if
an artifact digest no longer matches the passed quantization report.

When only the authoritative model card changes, validate and publish it without
scanning or re-uploading the runtime artifacts:

```bash
uv run python scripts/hugging_face/upload.py dots-tts --card-only --dry-run
uv run python scripts/hugging_face/upload.py dots-tts --card-only
```

When the local converted directory contains an upstream model card, exclude it
from the upload and publish this repo's own card instead.

The published subfolder should contain every runtime file required for loading
the artifact from a local clone of the Hugging Face repo.

## README Expectations

Each Hugging Face repo card should state:

- the upstream source model
- that the artifact is converted for MLX runtime use
- the published quantization variant
- the intended task or pipeline tag
- a minimal `mlx-speech` usage example
- any material license or usage restrictions
- a Links section pointing to the [source repo](https://github.com/appautomaton/mlx-speech)
  and the [project page](https://appautomaton.renocrypt.com/mlx-speech/)

## Release Checklist

Before uploading:

1. Verify the converted artifact loads from the local quantized path.
2. Run one short end-to-end inference or decode pass on that path.
3. Confirm the folder contains only the files intended for publication.
4. Confirm the Hugging Face repo card is present and accurate.
5. For dots.tts, run the required `--dry-run` gate and inspect all four remote
   paths before starting the resumable upload.

During upload:

1. Prefer the wrapper scripts in `scripts/hugging_face/`.
2. For large folders, prefer `hf upload-large-folder` so the upload can resume.
3. Do not run multiple uploads for the same folder in parallel.
4. Do not delete the local `.cache/huggingface/` upload state while an upload
   is active.

After upload:

1. Check the remote repo file list.
2. Confirm the quantized subfolder name is correct.
3. Confirm `model.safetensors` and all required config/tokenizer files landed.
4. Re-run one load test against the published file layout when practical.

## Operational Notes

- The upload wrappers are the canonical release entry points for this repo.
- When the standard `hf upload` path is unreliable for large files, use the
  resumable large-folder flow.
- If a model directory contains both published MLX artifacts and local
  reference material, stage only the publishable subfolder.
- Record meaningful artifact changes in the repo card or release notes when
  contents change.

## dots.tts Release Record — 2026-07-30

Published the four artifacts to `appautomaton/dots-tts-mlx` at revision
`0af7ad2f837278b364902500d086553f1586ce9a`. A README-only link correction
advanced the repository to revision
`5dde9ded6c577a84a71b5ee9dafebfa53188d6d6`; all 16 safetensors LFS SHA-256
values are unchanged between those revisions.

The authenticated remote inventory contained the Hub-generated
`.gitattributes`, the authoritative root `README.md`, and 56 runtime files under
exactly these prefixes:

- `soar/mlx-base/`
- `soar/mlx-int8/`
- `mf/mlx-base/`
- `mf/mlx-int8/`

No `original/`, `mlx-bf16/`, or sibling artifact path was present. The large
folder upload reported `56/56` files committed, 16.7 GB processed, and zero
ignored candidates.

Remote verification ran:

```bash
RUN_LOCAL_INTEGRATION=1 MLX_SPEECH_REQUIRE_CHECKPOINTS=1 \
  .venv/bin/python -m pytest -s tests/integration/test_dots_tts_hf.py
```

The runner additionally set a 16 GiB MLX memory limit. All four cases passed in
`988.95 s`: `dots-tts-soar-base`, `dots-tts-soar`, `dots-tts-mf-base`, and
`dots-tts-mf`. Each case used a new Hugging Face cache, materialized only its
selected artifact subtree and root README, strict-loaded the checkpoint, and
produced finite, non-silent mono 48 kHz continuation-clone waveform output.
Peak MLX allocation was `6,521,655,508` bytes; macOS process peak RSS was
`5,784,944,640` bytes. Generated/reference audio and downloaded caches remain
outside Git.

After the README-only correction, the same four isolated-cache cases passed
against revision `5dde9ded6c577a84a71b5ee9dafebfa53188d6d6` in `262.11 s`.
Peak MLX allocation was `6,521,655,508` bytes and macOS process peak RSS was
`6,423,871,488` bytes.

## mlx-speech 0.5.0 and dots.tts Card Refresh — 2026-08-03

Tag `v0.5.0` points to merge commit
`e80689094721d0c8d995139b7ac2f1defd5e2a16`. GitHub Actions run
[`30828268581`](https://github.com/appautomaton/mlx-speech/actions/runs/30828268581)
passed version consistency, the macOS unit suite, wheel-content inspection,
and trusted publishing to PyPI. No manual package upload was used.

PyPI published these immutable 0.5.0 files:

- `mlx_speech-0.5.0-py3-none-any.whl` at
  `2026-08-03T15:38:40.392338Z`, SHA-256
  `e8b6df537177c0e7d917f0adbd8b9338b949a9b9b365fdebcefa77e118ffe22a`
- `mlx_speech-0.5.0.tar.gz` at `2026-08-03T15:38:41.783955Z`, SHA-256
  `635dc243e4db52d50ba97b6b83ef9709e0f9e4f35a1202cb3d2eeda5c39ad4b3`

The public wheel was downloaded, matched its PyPI digest, passed the repository
wheel-content verifier, and installed into a clean virtual environment. That
installed package loaded the default `dots-tts-soar` alias from
`soar/mlx-int8` and streamed two finite, non-silent waveform chunks totaling
7,680 mono samples at 48 kHz.

After PyPI was live, the `--card-only` release path uploaded only the
authoritative root `README.md`. It advanced the Hugging Face repository to
revision `ca6740366ab7316c26ddbb640e756c4ec414778f`; no model artifact was
selected or re-uploaded. The published README exactly matched the repository
source at SHA-256
`bcdec6f61dc843bd1b5a95be181dbe26f8a3ee76f4beb40df4059bd11c7add31`
and documents `mlx-speech>=0.5.0`, speaker-only cloning, and bounded waveform
streaming.

## mlx-speech 0.5.1 and Granite Speech int8 — 2026-08-03

Pull request [#26](https://github.com/appautomaton/mlx-speech/pull/26) merged as
`b3202b7d5c61140d1125912170d86786c8a3d672`. Tag `v0.5.1` points to that
merge. GitHub Actions run
[`30848296791`](https://github.com/appautomaton/mlx-speech/actions/runs/30848296791)
passed version consistency, 914 macOS unit tests, wheel-content inspection,
build artifact upload, and trusted publishing to PyPI.

PyPI published these immutable 0.5.1 files:

- `mlx_speech-0.5.1-py3-none-any.whl` at
  `2026-08-03T20:04:18.727509Z`, SHA-256
  `c8d6f22207c9a237d6875d2586e40ac42fa82f24020b5a136c28a70b34e45a4f`
- `mlx_speech-0.5.1.tar.gz` at `2026-08-03T20:04:20.037794Z`, SHA-256
  `9f0eb36bc8bb0f14716e2521a5667917dc9a6f8ca11f1cdd4b61a17ad7641e6d`

The public `appautomaton/granite-4.0-1b-speech-int8-mlx` repository was
published at revision `797e3587f7353bb32bbe3e1ef75ee438672a51af`. Its
inventory contains the 12 staged artifact files, the authoritative root
`README.md`, and the Hub-generated `.gitattributes`. The remote
`model.safetensors` is 2,904,308,838 bytes with SHA-256
`cf355a69e931ccac95d5cf942c3d540ba2456f06ad89c379c8132875b9098e6c`.
The remote README and config Git blob IDs exactly match the release sources:
`d018fedf3356f7b593e656b7aeb878bdcb64a33a` and
`eaae63fb5f435c4938d9ec3a7c1bef692a180087`, respectively.

Remote verification installed `mlx-speech==0.5.1` from PyPI into an isolated
package directory and used a new unauthenticated Hugging Face cache. Loading
`granite-speech-4.0-1b` fetched the 11 allowed runtime files from the public
repository, reconstructed the saved `QuantizedLinear` module tree, and
transcribed the bundled sample with the required
`but timothy was a spoiled cat` prefix.
