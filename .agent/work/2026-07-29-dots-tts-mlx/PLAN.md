# dots.tts MLX Inference and Release Plan

## Goal

Implement the approved [dots.tts MLX capability](./SPEC.md) for SOAR and MeanFlow, validate source-faithful `mlx-base` and selective `mlx-int8` waveform inference, and publish the four runtime artifacts with one authoritative Hugging Face model card.

## Architecture Approach

The hard-to-reverse decisions are recorded in [DESIGN.md](./DESIGN.md): extract a shared internal Qwen2 trunk rather than depend on `mlx-lm`; convert once into an explicit MLX-native artifact contract; preserve FP32 for parity-sensitive CAM++ and AudioVAE encoding while using BF16 for the validated core/decoder paths; quantize only eligible Qwen weights; keep generation behind the existing non-streaming TTS adapter; and publish one family repository with four independently loadable subdirectories.

## Execution Routing and Topology

Default: direct, serial, and continue automatically after each slice verifies. Execution should continue through every approved slice; context-management windows are not planned stopping points.

Overrides:

- Slice 4: subagent recommended because extracting the Qwen2 trunk crosses the existing VibeVoice family and introduces a shared internal interface.
- Slice 9: subagent recommended because the revised per-path dtype schema crosses conversion, metadata, strict loading, and full-checkpoint parity.
- Slice 10: subagent recommended because the autoregressive runtime composes every model component and the unified TTS registry/API.

**Parallel-safe groups:** none. Component fixtures, `mlx-base` naming and dtype predicates, int8 packaging, and release claims form one dependency chain.

Checkpoints: none planned. Slice 14 proceeds under the publication authority in the approved spec; if Hugging Face credentials or `appautomaton` permissions are unavailable, execution must stop and report the required human action rather than weakening the release outcome.

## Requirement Traceability

| Requirement | Satisfying slices |
| --- | --- |
| REQ-001 — source acquisition and layout | 1, 9 |
| REQ-002 — complete pure-MLX waveform runtime | 4–10 |
| REQ-003 — unified public API | 10 |
| REQ-004 — explicit MLX conversion | 3, 9 |
| REQ-005 — component-validated base and int8 artifacts | 9, 11, 12 |
| REQ-006 — parity and behavioral validation | 2–12 |
| REQ-007 — Hugging Face release and card | 13, 14 |
| REQ-008 — documentation and verification integration | 1–14 |

## Ordered Slice Sequence

### Slice 1: Pin and audit official source snapshots

**Objective:** Add a reproducible acquisition/audit workflow and stage the pinned SOAR and MeanFlow originals without changing or tracking their contents.

**Acceptance criteria:**

- The workflow downloads only the SPEC-pinned Hugging Face revisions into `models/dots_tts/{soar,mf}/original/`.
- An audit records expected files, sizes, safetensors dtypes, source revisions, and the latent-statistics structure/hash.
- Both shallow reference checkouts and their pinned commits remain documented; weights remain ignored.

**Verification:** `uv run python scripts/convert/download_dots_tts.py --variant all --verify && pytest tests/unit/test_dots_tts_source_assets.py`

**Touches:** `scripts/convert/download_dots_tts.py`, `scripts/audit/dots_tts_source.py`, `tests/unit/test_dots_tts_source_assets.py`, `docs/references.md`, gitignored `models/dots_tts/`

**Produces:** Verified immutable SOAR and MeanFlow source directories plus a machine-readable source manifest.

**Status:** complete
**Evidence:** Added `scripts/convert/download_dots_tts.py`, `scripts/audit/dots_tts_source.py`, and `tests/unit/test_dots_tts_source_assets.py`; downloaded and hash-verified both pinned 4.8 GB snapshots with `uv run python scripts/convert/download_dots_tts.py --variant all --verify`; `5 passed`; Ruff passed; wrote gitignored `models/dots_tts/source_manifest.json` with exact file, safetensors dtype, and restricted latent-statistics inventories.
**Risks / next:** Slice 2 must consume this manifest as its provenance gate; the isolated PyTorch oracle remains the reviewed Apple Silicon resource risk.

### Slice 2: Generate and provenance-check official oracle fixtures

**Objective:** Produce the complete bounded numeric fixture pack from the pinned official PyTorch implementation before any MLX component code relies on it.

**Acceptance criteria:**

- A repository-owned audit wrapper launches the official checkout in an isolated oracle environment; Torch, Transformers, torchaudio, and the upstream package never enter the project environment, converter, published package, or runtime.
- Deterministic fixtures cover text scheduling, Qwen hidden/cache/EOS, latent IO, semantic prefill/decode, speaker fbank/embedding, AudioVAE encode/decode, SOAR and MeanFlow DiT steps, and both solver paths.
- The checked-in fixture pack contains only bounded numeric arrays and a manifest—no weights, reference audio, generated audio, or copied upstream code.
- The manifest records both reference commits, both official weight revisions, source hashes, oracle dependency versions, command, seed, input construction, tensor names/shapes/dtypes, tolerances, and fixture hashes.
- Regeneration into a temporary directory compares equal within recorded tolerances, and verification rejects any fixture whose provenance no longer matches Slice 1's source manifest.

**Verification:** `uv run python scripts/audit/dots_tts_oracle.py regenerate --variant all --compare tests/fixtures/dots_tts && pytest tests/unit/test_dots_tts_oracle_fixtures.py`

**Depends on:** Slice 1

**Touches:** `scripts/audit/dots_tts_oracle.py`, isolated oracle environment metadata, `tests/fixtures/dots_tts/`, `tests/unit/test_dots_tts_oracle_fixtures.py`

**Produces:** A reproducible, source-pinned official-oracle fixture pack that every component parity slice can consume without inventing its own oracle procedure.

**Status:** complete
**Evidence:** Added the isolated Python 3.12 oracle wrapper/worker and pinned dependency metadata under `scripts/audit/`, plus `tests/unit/test_dots_tts_oracle_fixtures.py`; captured 16 bounded NPZ fixtures and `tests/fixtures/dots_tts/manifest.json` for SOAR/MF; `uv run python scripts/audit/dots_tts_oracle.py regenerate --variant all --compare tests/fixtures/dots_tts` matched; `5 passed`; Ruff passed.
**Risks / next:** The project-local `.venv-torch` is intentionally unused because Python 3.13/Torch 2.11/Transformers 5 do not match the pinned official oracle; subsequent slices consume the checked-in arrays without Torch.

### Slice 3: Define config, text schedule, and checkpoint contract

**Objective:** Establish the dots.tts configuration, tokenizer/schedule behavior, native artifact schema, and strict metadata validation before model modules consume weights.

**Acceptance criteria:**

- Config parsing distinguishes SOAR flow matching from MeanFlow and preserves unknown upstream fields safely.
- Token IDs, language tags, prompt/text layout, audio spans, and generation budgets match pinned oracle fixtures.
- The native artifact contract requires configs, component safetensors, latent stats, tokenizer assets, source provenance, dtype policy, and quantization metadata.
- Invalid modes, inconsistent metadata, missing assets, and unsupported layouts fail explicitly.

**Verification:** `pytest tests/unit/test_dots_tts_config.py tests/unit/test_dots_tts_text.py tests/unit/test_dots_tts_checkpoint_contract.py`

**Depends on:** Slice 2

**Touches:** `src/mlx_speech/models/dots_tts/{config,text,checkpoint}.py`, `tests/unit/test_dots_tts_{config,text,checkpoint_contract}.py`, small oracle fixtures

**Produces:** A tested dots.tts config/schedule layer and an enforceable native checkpoint schema.

**Status:** complete
**Evidence:** Added `src/mlx_speech/models/dots_tts/{config,text,checkpoint}.py` and package exports; SOAR/MeanFlow mode parsing, Qwen/GQA validation, unknown-field preservation, official tts/interleave schedule parity, language tagging, exact artifact/tokenizer layout, provenance, dtype, quantization, and latent-statistics validation are covered; `15 passed`; Ruff passed.
**Risks / next:** The schema intentionally requires exact tokenizer assets and explicit int8 path metadata; Slice 4 must preserve VibeVoice behavior while moving Qwen2 math behind the new family-neutral contract.

### Slice 4: Extract and extend the shared Qwen2 trunk

**Objective:** Move the existing VibeVoice Qwen2 math into a family-neutral internal module and add the token/embed, KV-cache, tied-embedding, hidden-state, and EOS behavior required by dots.tts.

**Acceptance criteria:**

- Qwen2.5-1.5B config fields, GQA, RoPE offsets, RMSNorm, input IDs/input embeddings, and incremental cache behavior match pinned fixtures.
- dots.tts obtains final hidden states and EOS logits without constructing an unused language-model sampler.
- VibeVoice retains its behavior through a compatibility import and regression tests.
- No `mlx-lm`, Transformers, or Torch dependency is added.

**Verification:** `pytest tests/unit/test_dots_tts_qwen.py tests/unit/test_vibevoice_qwen2.py tests/test_vibevoice_generation.py`

**Execution:** subagent recommended

**Depends on:** Slice 3

**Touches:** shared internal Qwen2 module, `src/mlx_speech/models/vibevoice/qwen2.py`, dots.tts Qwen wrapper, corresponding tests

**Produces:** One shared pure-MLX Qwen2 trunk used by dots.tts and VibeVoice, plus compatibility/parity coverage.

**Execution correction:** The checked-in oracle pack intentionally contains no Qwen weights, so Slice 4 validates shared Qwen equations, token/embed paths, cache growth, tied output projection, EOS handling, and VibeVoice compatibility with deterministic tiny weights. Slice 9 must add the full native-BF16 Qwen comparison against `qwen.npz` after converted weights exist; this keeps unit tests checkpoint-free and does not weaken the pre-generation parity gate.

**Status:** complete
**Evidence:** Extracted `src/mlx_speech/models/_qwen2.py`, preserved VibeVoice through compatibility exports, and added `src/mlx_speech/models/dots_tts/qwen.py`; deterministic tests cover IDs/embeddings, GQA, caches, tied logits, EOS, and family-specific BF16 RoPE policy; focused suite `18 passed`, full unit suite `614 passed`, Ruff/forbidden-import/diff checks passed; spec and quality reviewers `APPROVED` after one BF16 compatibility correction. See `orchestration/slice-004-summary.md`.
**Risks / next:** Full official-weight `qwen.npz` parity is required in Slice 9 after native BF16 conversion; no implementation risk remains for the shared interface.

### Slice 5: Implement latent IO and the causal semantic encoder

**Objective:** Implement latent normalization and the causal semantic patch encoder, including incremental prefill/decode state used to feed generated patches back into Qwen.

**Acceptance criteria:**

- Latent mean/variance normalization and denormalization match oracle fixtures.
- Full-sequence and incremental semantic-encoder paths agree within checked-in tolerances.
- Patch sizing, causality, rotary/QK normalization, and cache boundaries reject invalid shapes.
- The module consumes MLX-native tensor layouts only.

**Verification:** `pytest tests/unit/test_dots_tts_latent_io.py tests/unit/test_dots_tts_semantic_encoder.py`

**Depends on:** Slice 3

**Touches:** `src/mlx_speech/models/dots_tts/{latent,semantic_encoder,layers}.py`, focused tests and fixtures

**Produces:** A cache-aware semantic feedback path with verified latent normalization.

**Execution correction:** Slice 5 compares latent normalization directly with `latent_io.npz` and validates semantic equations plus full/incremental cache equivalence with deterministic tiny weights. Full converted-weight semantic comparison against `semantic.npz` moves to Slice 9 alongside the already-recorded Qwen parity gate because the fixture pack contains no model weights.

**Status:** complete
**Evidence:** Added strict `LatentStatistics`/`LatentIO`, MLX-native causal convolution/semantic attention layers, and `VAESemanticEncoder` full, prefill, and patch-decode paths; latent normalization matches `latent_io.npz`, deterministic semantic tests cover causal full/cache equivalence, state shape, and invalid boundaries; `8 passed`; Ruff passed.
**Risks / next:** Fused full versus cached SDPA differs numerically by about 1.6e-3 but passes the recorded 0.02 semantic tolerance; Slice 9 retains the converted-weight `semantic.npz` gate.

### Slice 6: Implement CAM++ speaker conditioning

**Objective:** Implement the dots.tts audio front end and 512-dimensional CAM++ speaker encoder as a pure-MLX conditioning path.

**Acceptance criteria:**

- Mono loading, trimming, resampling, Kaldi-compatible filterbank extraction, mean normalization, and the 10-second cap match oracle fixtures.
- CAM++ layers and frozen batch-normalization behavior load native weights and reproduce the pinned speaker embedding tolerance.
- Speaker scale and projection produce the expected DiT conditioning shape.
- No torchaudio, librosa, ONNX runtime, or upstream package is required.

**Verification:** `pytest tests/unit/test_dots_tts_audio_frontend.py tests/unit/test_dots_tts_speaker.py`

**Depends on:** Slice 3

**Touches:** `src/mlx_speech/models/dots_tts/speaker.py`, shared audio helpers only where family-neutral, focused tests and fixtures

**Produces:** A pure-MLX reference-audio-to-speaker-conditioning pipeline.

**Execution correction:** Slice 6 reproduces the deterministic synthetic-audio resampling/fbank fixture now and validates CAM++ architecture, frozen batch normalization, cropping, scaling, and projection with deterministic tiny weights. Converted-weight comparison against the oracle speaker embedding/projected output moves to Slice 9 because `speaker.npz` contains no CAM++ weights.

**Status:** complete
**Evidence:** Added `speaker.py` with deterministic mono/cap handling, torchaudio-compatible default sinc resampling, Kaldi/Povey fbank + CMN, explicit frozen BatchNorm CAM++ layers, speaker scale, and projection; synthetic 48 kHz input matches the official 62×80 fbank fixture; tiny-model tests cover BatchNorm, embedding/projection shapes, scaling, determinism, and invalid inputs; `7 passed`; Ruff passed.
**Risks / next:** Full 512-dimensional CAM++ and projected numeric parity remains a Slice 9 converted-weight gate; the fbank/front-end path is already oracle-matched.

### Slice 7: Implement AudioVAE and causal BigVGAN waveform paths

**Objective:** Implement reference-audio latent encoding and 48 kHz causal waveform decoding, including alias-free activations and state needed for correct patch boundaries.

**Acceptance criteria:**

- Causal convolution, transposed convolution, SnakeBeta, resampling/filtering, encoder, latent sampling, and decoder modules match component fixtures.
- Vocoder weight normalization is absent from runtime modules because conversion folds it once.
- Reference encoding produces the correct continuous latent distribution and decoder output is finite, shaped correctly, and non-silent for a fixed latent fixture.
- Full and incremental internal decode paths agree where both are implemented; no public streaming API is added.

**Verification:** `pytest tests/unit/test_dots_tts_audio_vae.py tests/unit/test_dots_tts_vocoder.py`

**Depends on:** Slice 3

**Touches:** `src/mlx_speech/models/dots_tts/{audio_vae,vocoder,layers}.py`, focused tests and fixtures

**Produces:** Verified reference-latent encoding and 48 kHz waveform decoding modules.

**Execution correction:** Slice 7 validates the complete AudioVAE/BigVGAN architecture, MLX tensor layouts, causality, alias-free primitives, finite/non-silent waveform output, and deterministic full/incremental behavior with tiny weights. Converted-weight comparisons against `audio_vae.npz` move to Slice 9 because the fixture contains no vocoder weights.

**Status:** complete
**Evidence:** Added pure-MLX AudioVAE encoding/decoding, residual SLSTM bridge, causal/symmetric convolutions, causal transposed convolution, alias-free SnakeBeta filtering, AMP residual blocks, and BigVGAN waveform decoding. Focused tests cover exact causal boundaries, stride-length preservation, deterministic buffered decode equivalence, latent/waveform contracts, and finite non-silent output; `8 passed`; Ruff passed.
**Risks / next:** Full-size converted-weight AudioVAE/vocoder numeric parity remains a Slice 9 gate; runtime modules already use folded convolution weights and contain no weight-normalization parametrization.

### Slice 8: Implement the DiT, SOAR solver, and MeanFlow solver

**Objective:** Implement the shared adaptive-normalization DiT and both approved next-patch solvers with their distinct guidance semantics.

**Acceptance criteria:**

- DiT projections, attention masks/positions, prefix-history layout, timestep embeddings, speaker conditioning, and output extraction match fixtures.
- SOAR supports Euler flow matching and classifier-free guidance with the approved defaults.
- MeanFlow uses its duration embedding, four evaluations by default, and no runtime CFG.
- Fixed seeds reproduce the same normalized patch within each artifact.

**Verification:** `pytest tests/unit/test_dots_tts_dit.py tests/unit/test_dots_tts_solvers.py`

**Depends on:** Slices 4 and 5

**Touches:** `src/mlx_speech/models/dots_tts/{dit,solvers}.py`, focused tests and fixtures

**Produces:** A shared DiT with separately verified SOAR and MeanFlow next-patch solvers.

**Execution correction:** Slice 8 validates the shared DiT architecture, attention/mask/position behavior, adaptive conditioning, prefix-tail coordinate splicing, and both solver equations/defaults with deterministic tiny weights. Converted-weight comparisons against `dit.npz` and `solver.npz` move to Slice 9 because those fixtures contain no DiT weights.

**Status:** complete
**Evidence:** Added a shared pure-MLX DiT with official cos-first timestep embeddings, optional MeanFlow duration embedding, affine-free adaLN blocks, RMS-normalized rotary attention, mask/position handling, speaker conditioning, and tail output extraction. Added separate SOAR and MeanFlow solvers: SOAR performs 10-step Euler integration by default with batched CFG at guidance 1.2; MeanFlow performs four single-branch duration-conditioned evaluations by default with no runtime CFG. Focused deterministic tests cover causality, positions, conditioning, prefix splicing, solver schedules/equations, seed repeatability, and validation; `10 passed`; Ruff passed.
**Risks / next:** Official full-size `dit.npz` and `solver.npz` parity requires converted DiT/core weights and remains a blocking Slice 9 load gate for both variants.

### Slice 9: Build `mlx-base` conversion and strict mixed-precision loading

**Objective:** Convert both official checkpoints into self-contained MLX-native `mlx-base` artifacts whose path-specific dtypes match the revised source-faithful contract, then strict-load and parity-check every component.

**Acceptance criteria:**

- Conversion remaps keys, transposes convolutions, folds all 80 vocoder weight-normalization pairs, and writes only MLX-native layouts.
- `core.safetensors` is BF16; `speaker.safetensors` is FP32; `latent_stats.safetensors` is FP32; `vocoder.safetensors` stores `audio_encoder.*`, `enc_mi_layer.*`, and `pre_proj.*` as FP32 and its decode paths as BF16.
- `mlx_config.json` names artifact class `base` and serializes exact component/path dtype predicates; strict loading rejects wrong-dtype tensors as well as duplicate, missing, unexpected, mismatched, or source-shaped tensors.
- The restricted latent-statistics reader accepts only the pinned NumPy pickle structure and emits Torch-free safetensors.
- Alignment reports account for every source tensor and explicitly name dropped training-only buffers.
- SOAR and MeanFlow `mlx-base` directories strict-load and pass all checked-in Qwen, semantic, speaker, AudioVAE encode/decode, DiT, and solver fixture tolerances. The audit documents the official oracle's FP32 comparison semantics separately from stored runtime dtype.
- Obsolete local `mlx-bf16` directories are never accepted as `mlx-base` inputs or release targets.

**Verification:** `pytest tests/unit/test_dots_tts_convert.py tests/unit/test_dots_tts_checkpoint.py tests/unit/test_dots_tts_speaker.py tests/unit/test_dots_tts_audio_vae.py && pytest tests/checkpoint/test_dots_tts_base_load.py && uv run python scripts/audit/dots_tts_checkpoint.py --variant all --precision base`

**Execution:** subagent recommended

**Depends on:** Slices 4–8

**Touches:** `scripts/convert/dots_tts.py`, `scripts/audit/dots_tts_checkpoint.py`, dots.tts checkpoint/loading/precision metadata, focused unit and checkpoint tests, gitignored `mlx-base` artifacts

**Produces:** Strict-loadable, fixture-verified SOAR and MeanFlow `mlx-base` artifacts plus conversion/alignment reports.

**Replan context:** The first Slice 9 attempt proved native mapping and strict shape alignment, corrected CAM++ length-aware statistics, and empirically rejected all-BF16 storage. Execution resumes from that work; it must replace the old dtype schema and finish the mixed-precision parity gate rather than redo Slices 1–8.

**Status:** complete
**Evidence:** Added transactional, source-integrity-verified `mlx-base` conversion and strict mixed-precision loading for SOAR and MeanFlow, with exact path dtype predicates, Torch-free latent-stat conversion, complete source accounting, and encoder-only tiled FP32 reductions. Full unit tier `675 passed`; real checkpoint tier `2 passed`; exact all-variant audit passed with AudioVAE encode max error `0.000179`, waveform `0.001367`, SOAR solver `0.015291`, and MeanFlow solver `0.007935`. A representative 3.2-second SOAR encode completed in `1.995 s` with `1,640,644,608` incremental Metal bytes and a `25,165,824`-byte largest logical reduction tile. Scoped Ruff, forbidden-import scan, and `git diff --check` passed; spec and quality reviewers `APPROVED` after one bounded integrity, rollback, latent-shape, and memory correction round. See `orchestration/slice-009-summary.md`.
**Risks / next:** Host- and workload-specific end-to-end peak memory remains a Slice 10 generation/integration concern; Slice 10 must compose waveform generation without weakening the strict `mlx-base` dtype and loader contract.

### Slice 10: Compose autoregressive generation and the unified TTS adapter

**Objective:** Connect prompt conditioning, Qwen, both solvers, semantic feedback, EOS, and AudioVAE decoding into `mlx-base` waveform generation exposed through `mlx_speech.tts`.

**Acceptance criteria:**

- Continuation cloning, speaker-only cloning, and documented no-reference parity behavior follow the official schedule semantics.
- SOAR and MeanFlow generate finite, non-silent mono 48 kHz waveform through `TTSOutput` with deterministic seeds and bounded patch generation.
- `dots-tts-soar-base` and `dots-tts-mf-base` aliases resolve correctly; short int8 aliases remain unpublished or resolve to base until the quant gate passes.
- Hub alias metadata supports an explicit nested artifact subdirectory while preserving every existing flat-repository alias unchanged.
- Each remote dots alias passes exactly `["<artifact-subdir>/**", "README.md"]` as its download allow-list and returns that explicit subdirectory; it never falls back to quantization-directory guessing.
- A mocked resolver test and an isolated-cache integration probe prove that resolving one alias materializes no `.safetensors` from the other three variants.
- Language, solver, guidance, speaker, seed, and patch-budget controls stay adapter kwargs.
- Dependency guards prove the runtime does not import forbidden packages or `.references/` code.

**Verification:** `pytest tests/unit/test_dots_tts_generation.py tests/unit/test_dots_tts_adapter.py tests/unit/test_dots_tts_dependency_guard.py tests/unit/test_dots_tts_hub_selective.py tests/unit/test_hub_snapshot_resolve.py && pytest tests/runtime/test_dots_tts_base.py && RUN_LOCAL_INTEGRATION=1 pytest tests/integration/test_dots_tts_base.py`

**Execution:** subagent recommended

**Depends on:** Slice 9

**Touches:** `src/mlx_speech/generation/dots_tts.py`, `src/mlx_speech/tts/_adapters/dots_tts.py`, TTS registry/hub aliases, model composition, tests

**Produces:** End-to-end `mlx-base` dots.tts generation through the unified public API.

**Status:** complete
**Evidence:** Added pure-MLX autoregressive generation and the unified dots.tts adapter, explicit nested Hugging Face artifact aliases, dependency/isolation guards, and focused runtime/integration coverage. Replaced dense all-phase resampling with a source-faithful output-centered Kaiser-sinc implementation bounded to a 32 MiB workspace and 256 MiB output, with early continuation-budget rejection and speaker-only prefix limiting. Focused suite `50 passed`; full unit tier `717 passed`; real SOAR/MF runtime and unified-API integration passed independently; SOAR 44.1 kHz speaker-only and continuation conditioning passed. Peak observed RSS was about 4.85 GiB and peak MLX allocation about 5.24 GiB under explicit stop guards. Ruff and scoped diff checks passed; spec and quality reviewers `APPROVED`. See `orchestration/slice-010-summary.md`.
**Risks / next:** Long multi-patch real-checkpoint trajectories remain a Slice 12 quality/integration concern; Slice 11 may now build selective Qwen-only int8 artifacts without changing the verified base runtime.

### Slice 11: Produce and load selective int8 artifacts

**Objective:** Quantize the approved Qwen2.5 predicate for both checkpoints, serialize exact metadata, and strict-load `mlx-int8` while preserving every non-selected `mlx-base` dtype.

**Acceptance criteria:**

- Affine int8 group-size 64 is applied to eligible Qwen Linear/Embedding paths and reconstructed from serialized path-aware metadata.
- CAM++, AudioVAE encoding, latent statistics, semantic/flow paths, conditioning projections, and waveform decoding retain their exact `mlx-base` dtypes; no non-Qwen path is quantized.
- SOAR and MeanFlow int8 directories are self-contained and at least 25% smaller than their matching `mlx-base` artifacts.
- Quantized alignment reports and checkpoint tests have no unexplained gaps.

**Verification:** `pytest tests/unit/test_dots_tts_quantization.py tests/unit/test_dots_tts_checkpoint.py && pytest tests/checkpoint/test_dots_tts_int8_load.py && uv run python scripts/audit/dots_tts_checkpoint.py --variant all --precision int8`

**Depends on:** Slice 10

**Touches:** dots.tts checkpoint/quantization code, converter presets, audit script, tests, gitignored int8 artifacts

**Produces:** Strict-loadable SOAR and MeanFlow `mlx-int8` artifacts with serialized selective-quantization metadata.

**Status:** complete
**Evidence:** Added exact native `qwen.model.*` affine-int8 reconstruction for every eligible Qwen Linear/Embedding module, strict packed-weight/scale/bias dtype and path validation, transactional base-to-int8 conversion, inherited-file hash checks, size enforcement, and int8 audit support. Both variants quantize 197 Qwen modules; SOAR is `4,893,076,106 → 3,446,414,994` bytes (`29.565%` smaller) and MeanFlow is `4,895,702,230 → 3,449,041,102` bytes (`29.550%` smaller). Focused suite `36 passed`, final full unit tier `722 passed`, real int8 checkpoint tier `2 passed`, and the all-variant component audit passed with Qwen minimum cosine `0.999023`; every non-Qwen metric retained its base tolerance. Peak observed conversion MLX allocation was about 4.60 GiB and strict-load peak about 3.20 GiB under explicit stop guards. Ruff and `git diff --check` passed.
**Risks / next:** Component parity does not authorize default aliases or release claims. Slice 12 must run continuation and speaker-only waveform integration plus the fixed-corpus WER, speaker-similarity, size, and peak-memory gate before int8 may become the default.

### Slice 12: Run the four-artifact quality and integration gate

**Objective:** Prove `mlx-base` correctness and decide whether int8 may become the default using the fixed multilingual cloning corpus and all four runtime artifacts.

**Acceptance criteria:**

- The eval corpus acquisition/manifest is reproducible and keeps source/generated audio gitignored.
- SOAR/MF × base/int8 each pass continuation and speaker-only waveform integration.
- Aggregate int8 WER regresses by no more than 1 absolute point and speaker cosine by no more than 0.02 against matching base; size and peak-memory measurements are recorded.
- A checked-in benchmark report records prompts, source revisions, artifact hashes, host, commands, metrics, failures, and whether int8 earned default status.
- Default int8 aliases are enabled only on PASS.

**Verification:** `uv run python scripts/eval/dots_tts_quant_gate.py --model-root models/dots_tts && RUN_LOCAL_INTEGRATION=1 pytest tests/integration/test_dots_tts.py && pytest tests/unit/ tests/checkpoint/ tests/runtime/`

**Depends on:** Slice 11

**Touches:** `scripts/eval/dots_tts_quant_gate.py`, eval manifest/acquisition support, integration tests, registry defaults, `docs/benchmarks/dots-tts-quant-gate-2026-*.md`

**Produces:** A reproducible four-artifact quality report and the evidence-backed default-alias decision.

**Status:** complete
**Evidence:** Added a reproducible macOS-built-in English/Mandarin corpus manifest and integrity-recorded materializer, a resumable four-artifact quantization gate, language-aware ASR error scoring, speaker-cosine/size/peak-memory measurements, full continuation and speaker-only integration coverage, and the checked-in report `docs/benchmarks/dots-tts-quant-gate-2026-07-30.md`. All eight `SOAR/MF × base/int8 × clone-mode` integration cases passed. Aggregate base and int8 WER were both `0.0294` (`+0.0000` regression); aggregate speaker cosine was `0.7930` base versus `0.8024` int8 (`-0.0094` regression). SOAR and MeanFlow each passed independently, so short aliases now select explicit `mlx-int8` subdirectories while `*-base` aliases remain fixed. Base generation peaked at `8.52 GiB` MLX versus `7.18 GiB` int8 in the quality run. The repository gate completed with `793 passed, 34 skipped`; the final unit tier completed with `728 passed`. Ruff and `git diff --check` passed.
**Risks / next:** The fixed gate contains two clean macOS synthesized speakers and one target sentence per language; it is a deterministic regression surface, not a broad subjective-quality benchmark. Slice 13 must state that limitation and use only the reproduced metrics above in documentation and the model card.

### Slice 13: Prepare documentation, model card, and release staging

**Objective:** Create the evidence-backed user guide, authoritative Hugging Face card, and safe resumable upload targets without publishing yet.

**Acceptance criteria:**

- `docs/dots-tts.md` covers checkpoint selection, clone modes, controls, layout, conversion, quantization, memory, limitations, and safety.
- The model card covers the four variants, exact upstream revisions, Apache-2.0 attribution, reproduced metrics only, usage examples, limitations, consent, disclosure, and misuse risk.
- Release tooling stages only `soar/mf × mlx-base/mlx-int8` plus the card; `original/` and obsolete `mlx-bf16/` directories are structurally unreachable from upload targets.
- A dry run shows the intended remote paths under `appautomaton/dots-tts-mlx` and fails if any required artifact or benchmark evidence is missing.

**Verification:** `uv run python scripts/hugging_face/upload.py dots-tts --dry-run && pytest tests/unit/test_dots_tts_release.py && git diff --check`

**Depends on:** Slice 12

**Touches:** `docs/dots-tts.md`, `docs/huggingface-release.md`, `scripts/hugging_face/upload.py`, `scripts/hugging_face/model_cards/appautomaton/dots-tts-mlx.md`, release tests

**Content constraints:** artifact targets are the model card and technical guide; sources are the pinned official/community repositories, approved SPEC/DESIGN, and locally reproduced benchmark report; factual risk is high; voice is technical and evidence-first with limitations stated plainly; unsupported upstream-parity, lossless-quantization, or real-time claims are prohibited.

**Produces:** The user guide, authoritative model card, safe release registry, and verified dry-run manifest.

### Slice 14: Publish and verify the Hugging Face release

**Objective:** Upload the four validated artifacts and authoritative card, then prove remote resolution and waveform generation.

**Acceptance criteria:**

- `appautomaton/dots-tts-mlx` contains exactly the approved card and four runtime subdirectories, with original checkpoints absent.
- Each remote variant resolves through its intended alias/repo-subdirectory mapping and strict-loads.
- Each alias is tested from a clean isolated Hugging Face cache; only its selected artifact subtree and root README may be materialized, and sibling-variant safetensors fail the gate.
- A short continuation-clone smoke produces finite, non-silent 48 kHz waveform for every remote variant.
- Published file lists, revisions, smoke commands, and results are recorded without adding weights or audio to Git.

**Verification:** `uv run python scripts/hugging_face/upload.py dots-tts && RUN_LOCAL_INTEGRATION=1 pytest tests/integration/test_dots_tts_hf.py`

**Depends on:** Slice 13

**Touches:** external `appautomaton/dots-tts-mlx`, post-upload integration test/evidence, release documentation if remote revisions must be recorded

**Produces:** The published four-variant Hugging Face family repository and recorded remote smoke evidence.

## Aggregate Verification Commands

| Gate | Command |
| --- | --- |
| Default development | `.venv/bin/python -m pytest tests/unit/` |
| Oracle fixture provenance | `uv run python scripts/audit/dots_tts_oracle.py regenerate --variant all --compare tests/fixtures/dots_tts` |
| Checkpoint contract/conversion | `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/` |
| Forward/generation/runtime | `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/ tests/runtime/` |
| End-to-end local waveform | `RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/test_dots_tts.py` |
| Quant quality | `uv run python scripts/eval/dots_tts_quant_gate.py --model-root models/dots_tts` |
| Release staging | `uv run python scripts/hugging_face/upload.py dots-tts --dry-run` |
| Remote waveform | `RUN_LOCAL_INTEGRATION=1 .venv/bin/python -m pytest tests/integration/test_dots_tts_hf.py` |

## Review: Engineering

- Verdict: approved_with_risks
- Strength: The revised plan makes each precision reduction evidence-gated and keeps conversion, native storage, strict loading, runtime composition, quantization, and publication in one traceable dependency chain.
- Concern: Slice 9 currently writes multi-gigabyte files directly into the final artifact directory, so interruption can leave a non-empty partial `mlx-base` directory that the converter refuses to resume or replace.
- Concern: Slice 9 may still fail FP32 AudioVAE encoder parity because preserving FP32 weights does not prevent reduced-precision MLX convolution accumulation, although the full encoder fixture gate will expose this before generation.
- Concern: Slice 9's oracle-comparison audit currently casts whole modules to FP32, which can hide a wrong runtime dtype or a BF16 decoder regression unless stored-dtype validation and runtime-dtype execution are tested separately.
- Concern: Slice 11 must serialize converted native `qwen.*` quantization paths and inherit every non-Qwen base dtype, while the current checkpoint schema still assumes the stale upstream `llm.*` prefix and uniform per-component dtypes.
- Action: In Slice 9, stage and atomically promote complete artifacts, implement a total native-path dtype predicate, and separate oracle-comparison from runtime-dtype gates; in Slice 11, assert that only native Qwen paths are quantized and all other tensor dtypes equal `mlx-base`.
- Verified: Traced the revised PLAN/DESIGN critical path against the current converter, checkpoint loader, audit implementation, native parameter namespaces, partial-output behavior, mixed-dtype binding, and specified unit/checkpoint/runtime gates.
