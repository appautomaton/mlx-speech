# dots.tts MLX Inference and Release

**Supersedes:** the 2026-07-29 revision of this spec, whose all-BF16 baseline was disproved during converted-weight parity testing.

**Bet:** A component-validated MLX port of dots.tts can provide high-quality local voice cloning on Apple Silicon without forcing precision-sensitive modules into a smaller dtype they cannot safely support.

## Bounded Goal

Add end-to-end dots.tts SOAR and MeanFlow inference to `mlx-speech`, producing 48 kHz waveform output through the unified TTS API from a source-faithful `mlx-base` artifact and a selectively quantized `mlx-int8` artifact, then publish only variants that pass their component and end-to-end quality gates under the `appautomaton` Hugging Face organization.

## Classification

- Work scale: capability
- Work shape: mixed — reference parity, model-family feature, checkpoint conversion, selective quantization, and release packaging
- Selected lenses: product, engineering, runtime, security

## Target Stakeholder

Apple Silicon developers using `mlx-speech` who want local multilingual dots.tts synthesis and authorized voice cloning without installing or importing PyTorch, Transformers, `mlx-lm`, or the upstream runtime.

## Broader Intent

This change advances `mlx-speech` as a clean multi-family speech library while making precision claims evidence-based. Artifact names, metadata, model cards, and default aliases must describe the precision actually used rather than implying blanket quantization.

## Approved Approach and Evidence

The approved checkpoint scope remains **SOAR + MeanFlow**:

- Official PyTorch reference: `.references/dots.tts` commit `5ed719e3d36f5a3f6d8037ca9a7009d4fd0520ba` (`v0.2.1`).
- Community MLX comparison: `.references/dots-tts-mlx` commit `f64479f51a2a9d7093533732cae86e765d8fb96e` (`v0.7.0`).
- Official SOAR weights: Hugging Face revision `e3520f75254d0020a0406db31c51a79d00d22d55`.
- Official MeanFlow weights: Hugging Face revision `25c53fb462e57087e52237daa5ea30df1c5cc328`.
- The official core checkpoint is BF16; the official vocoder and CAM++ checkpoints are FP32.

Converted-weight evidence established the revised precision policy:

- Both native SOAR and MeanFlow mappings strict-load with zero missing, unexpected, shape-mismatched, or source-shaped runtime tensors.
- Qwen matches the official fixture when compared under the oracle's FP32 compute semantics; its converted storage remains BF16.
- CAM++ matches the official oracle at FP32 with `2.1e-5` maximum error. Casting it to BF16 fails the checked-in speaker tolerance (`0.295` maximum error, cosine `0.99668`). CAM++ therefore remains FP32.
- The AudioVAE decoder converted to BF16 passes waveform parity (`0.00136` maximum error against a `0.02` tolerance).
- The AudioVAE reference encoder converted to BF16 fails latent-distribution parity (`2.76` maximum error against a `0.02` tolerance). Its encoder, encoder bridge, and distribution projection therefore remain FP32 unless a later, separately approved predicate passes the same gates.

This evidence proves the all-BF16 policy is invalid. It does not yet prove full end-to-end quality for the revised mixed-precision artifact; that remains a required gate.

The user selected `mlx-base` as the truthful name for the unquantized artifact. Exact component dtypes live in machine-readable metadata. `mlx-int8` means that the approved Qwen predicate is int8; it does not imply that every model tensor is int8.

## Required Outcome

### REQ-001 — Reproducible source acquisition and local layout

- Preserve the official Hugging Face snapshots unchanged under:
  - `models/dots_tts/soar/original/`
  - `models/dots_tts/mf/original/`
- Provide a reproducible, revision-pinned acquisition command or script.
- Keep original and converted weights gitignored; no model tensor enters Git history.
- Produce converted artifacts under each variant's `mlx-base/` and `mlx-int8/` directories.

### REQ-002 — Complete pure-MLX waveform runtime

Implement the complete inference path needed for waveform output:

- tokenizer and generation-schedule preparation;
- Qwen2.5-1.5B contextual trunk with incremental KV cache and EOS projection;
- causal semantic patch encoder;
- SOAR flow-matching DiT solver with classifier-free guidance;
- MeanFlow four-evaluation solver without runtime CFG;
- CAM++ speaker encoder and speaker projection;
- AudioVAE reference encoder and causal BigVGAN-style decoder;
- latent normalization, autoregressive feedback, stopping, and deterministic seed handling.

The runtime must use MLX for model computation and must not import or transitively require PyTorch, torchaudio, Transformers, `mlx-lm`, or upstream/reference packages.

### REQ-003 — Unified public API

- `mlx_speech.tts.load(...)` must recognize dots.tts checkpoints and return the existing `TTSModel` interface.
- Required aliases:
  - `dots-tts-soar` → `mlx-int8` only after the int8 quality gate passes;
  - `dots-tts-soar-base` → source-faithful mixed-precision baseline;
  - `dots-tts-mf` → `mlx-int8` only after the int8 quality gate passes;
  - `dots-tts-mf-base` → source-faithful mixed-precision baseline.
- Until int8 earns default status, the short aliases remain unpublished or resolve to `mlx-base`; they must never silently select an unvalidated int8 artifact.
- `generate(text, reference_audio=..., reference_text=...)` must support continuation voice cloning.
- `generate(text, reference_audio=...)` must support speaker-embedding-only cloning.
- The no-reference path may remain callable for upstream parity, but documentation must state that random-voice generation is not a quality-supported use of the released multi-speaker checkpoints.
- Model-specific controls remain backend kwargs: explicit language code, solver steps, guidance scale, speaker scale, seed, and maximum audio patches.

### REQ-004 — Explicit MLX checkpoint conversion

- Conversion remains separate from runtime loading.
- Convert PyTorch-shaped tensors into explicit MLX-native layouts, including convolution transposes and one-time vocoder weight-normalization folding.
- Convert `latent_stats.pt` into a Torch-free safetensors asset using a restricted reader that permits only the pinned NumPy structure.
- Preserve required tokenizer/configuration assets and record model family, checkpoint mode, source revision, artifact class, component dtype policy, quantization predicate, group size, and exact quantized paths in machine-readable metadata.
- Strict loading rejects missing, unexpected, duplicate, shape-mismatched, source-shaped, wrong-dtype, or metadata-inconsistent tensors.
- Conversion and alignment reports account for every source tensor; intentionally dropped training-only buffers are named explicitly.

### REQ-005 — Component-validated precision policy

The `mlx-base` artifact is the source-faithful correctness baseline:

| Component | Baseline storage | Rationale |
| --- | --- | --- |
| Qwen2.5 trunk, EOS head | BF16 | Official core source precision |
| Semantic encoder | BF16 | Official core source precision |
| DiT, SOAR/MeanFlow solver projections | BF16 | Official core source precision |
| Small conditioning projections | BF16 | Official core source precision |
| AudioVAE decoder and BigVGAN | BF16 | Converted waveform parity passed |
| CAM++ speaker encoder | FP32 | BF16 speaker parity failed |
| AudioVAE encoder, encoder bridge, distribution projection | FP32 | BF16 latent parity failed |
| Latent statistics | FP32 | Official statistics precision |

- The `mlx-int8` artifact applies affine 8-bit, group size 64, only to eligible Qwen Linear/Embedding paths that pass component and end-to-end gates; activations remain BF16.
- Every component not selected by the int8 predicate retains its `mlx-base` dtype. In particular, CAM++ and AudioVAE reference encoding remain FP32.
- Any future dtype reduction requires both component fixture parity and end-to-end waveform quality evidence. A size win alone never authorizes quantization.
- The exact dtype and quantization predicate is serialized per component/path and reconstructed automatically by the loader.
- Artifact names and documentation must not describe `mlx-base` as an all-BF16 model or `mlx-int8` as an all-int8 model.

### REQ-006 — Reference parity and behavioral validation

- Use the pinned official PyTorch source as the behavioral oracle and the pinned community MLX port as an implementation comparison, never as an imported runtime dependency.
- Add focused parity fixtures/tests for schedule construction, Qwen hidden/KV behavior, speaker features and embeddings, AudioVAE encode/decode, semantic feedback, DiT/MeanFlow patch generation, weight mapping, and EOS behavior.
- The base artifact must pass checked-in component tolerances for both SOAR and MeanFlow before generation integration proceeds.
- Add end-to-end integration coverage for SOAR and MeanFlow, `mlx-base` and `mlx-int8`, producing finite, non-silent 48 kHz waveform output.
- Evaluate int8 against the matching base artifact on a fixed multilingual voice-cloning corpus. Aggregate ASR WER may regress by no more than 1 absolute percentage point, and speaker-similarity cosine may regress by no more than `0.02`.
- Any weaker result blocks the int8 default and publication claim. Do not describe quantization as lossless unless recorded evaluation supports it.

### REQ-007 — Hugging Face release and model card

- Publish validated artifacts under `appautomaton/dots-tts-mlx` using the repository's resumable upload workflow.
- The family layout is:

  ```text
  appautomaton/dots-tts-mlx/
    README.md
    soar/mlx-base/
    soar/mlx-int8/
    mf/mlx-base/
    mf/mlx-int8/
  ```

- Original upstream checkpoints must not be included in the MLX release repository.
- Upload targets address only the four converted directories and the card; `original/` must be structurally unreachable.
- The model card includes upstream sources and revisions, Apache-2.0 terms, architecture, supported languages, exact component precision matrix, exact int8 scope, measured memory and quality, usage, known limitations, and voice-cloning consent/misuse guidance.
- Each published subdirectory loads by Hugging Face repo plus explicit subdirectory and completes an end-to-end waveform smoke test after upload.

### REQ-008 — Documentation and verification integration

- Add a dots.tts guide covering checkpoint choice, cloning modes, controls, layout, conversion, selective quantization, component precision, memory, and limitations.
- Update the model registry and Hugging Face release documentation without introducing dots-specific parameters into unrelated adapters.
- Add unit, checkpoint, runtime, and opt-in integration tests at the repository's established tiers.

## Acceptance Criteria

1. Revision-pinned acquisition reconstructs both unchanged `original/` directories, and conversion reconstructs SOAR/MF × `mlx-base`/`mlx-int8` without modifying them.
2. All four converted directories are self-contained, strict-load by local path, contain only Torch-free runtime assets, and record source revision plus exact component/path precision.
3. Every source tensor is consumed by a named MLX module or explicitly rejected; conversion and load reports contain no unexplained gaps.
4. Both `mlx-base` artifacts pass the checked-in component fixture tolerances, including FP32 CAM++, FP32 AudioVAE reference encoding, and BF16 AudioVAE waveform decoding.
5. Each of the four artifacts completes continuation and speaker-only cloning and returns a finite, non-silent mono `mx.array` at 48,000 Hz through `mlx_speech.tts.load(...).generate(...)`.
6. Fixed-seed generation is repeatable for the same artifact and inputs, and SOAR and MeanFlow use their intended solver semantics and defaults.
7. Runtime dependency guards show that importing and generating with dots.tts does not import PyTorch, torchaudio, Transformers, `mlx-lm`, or either `.references/` checkout.
8. Each `mlx-int8` artifact is at least 25% smaller than its matching `mlx-base` artifact and meets the WER and speaker-similarity regression limits in REQ-006.
9. Short aliases select int8 only after criterion 8 passes; base aliases always resolve to the source-faithful artifacts.
10. `pytest tests/unit/`, relevant checkpoint/runtime tiers, and opt-in four-artifact waveform integration tests pass before publication.
11. The Hugging Face repository contains the four approved runtime subdirectories and authoritative model card, excludes originals, resolves each artifact selectively, and passes post-upload waveform smokes.

## Constraints and Risks

- dots.tts is continuously autoregressive; memory grows with prompt and generated length even when Qwen is quantized.
- Mixed component dtypes increase metadata and strict-loading complexity. Wrong-dtype tensors must fail before inference.
- CAM++ and AudioVAE reference encoding are precision-sensitive. Their FP32 policy increases artifact size but protects speaker identity and continuation fidelity.
- The upstream and community runtimes contain optimization and caching choices that require independent parity proof before adoption.
- Reference audio processing, convolution layout, weight normalization, latent scaling, and per-path dtype assignment are high-risk parity boundaries.
- High-fidelity voice cloning creates impersonation and consent risks. Documentation and model cards must require authorized use and synthetic-audio disclosure.
- Publishing is an authorized external state change, but execution stops at a human-action checkpoint if credentials or organization permissions are unavailable.

## Scope Coverage

### Included

- SOAR and MeanFlow inference.
- Source-faithful `mlx-base` and selectively quantized `mlx-int8` artifacts.
- Continuation and speaker-only voice cloning with 48 kHz waveform output.
- Local conversion, strict dtype validation, quantization, parity, integration tests, aliases, documentation, Hugging Face packaging, model card, upload, and remote smoke validation.

### Deferred / Not in Scope

- `dots.tts-base`: SOAR supersedes it for the quality-focused released use case.
- Additional int8 components: deferred until a separately recorded predicate passes both component and end-to-end gates.
- MLX training, fine-tuning, SOAR alignment, or MeanFlow distillation: the MLX target is inference-only.
- A new public streaming TTS protocol: internal incremental state may be implemented, but the first public surface remains non-streaming.
- Persistent enrolled-speaker profiles, long-text sentence stitching, Swift/iOS support, automatic language detection, and dependency-heavy text normalization.

### Resolved Decision

- The source-faithful unquantized artifact is named `mlx-base`, not `mlx-bf16` or `mlx-mixed`. Exact component dtypes are authoritative in metadata and documentation.

## Anti-Goals

- Do not vendor, import, or package code from `.references/` as runtime code.
- Do not label a Torch-, Transformers-, or `mlx-lm`-backed path as MLX support.
- Do not stop at latent/token generation; end-to-end means waveform output.
- Do not force FP32 source components into BF16 or int8 after they fail their parity gate.
- Do not call `mlx-base` an all-BF16 artifact or `mlx-int8` an all-int8 artifact.
- Do not mutate or republish official source checkpoints as MLX artifacts.
- Do not commit model weights, reference audio, or generated evaluation audio.
- Do not broaden the shared TTS protocol with dots-specific controls.
- Do not claim upstream benchmark parity, lossless quantization, or real-time performance without reproduced evidence.
