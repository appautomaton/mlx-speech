# dots.tts MLX Inference and Release

**Bet:** A clean, adapter-based MLX port of dots.tts SOAR and MeanFlow can provide high-quality local voice cloning on Apple Silicon while preserving a Torch-free runtime and a smaller int8 default artifact.

## Bounded Goal

Add end-to-end dots.tts SOAR and MeanFlow inference to `mlx-speech`, producing 48 kHz waveform output through the unified TTS API from both BF16 and mixed W8A-BF16 checkpoints, then publish the validated runtime artifacts and model card under the `appautomaton` Hugging Face organization.

## Classification

- Work scale: capability
- Work shape: mixed — reference parity, model-family feature, checkpoint conversion, quantization, and release packaging
- Selected lenses: product, engineering, runtime, security

## Target Stakeholder

Apple Silicon developers using `mlx-speech` who want local multilingual dots.tts synthesis and authorized voice cloning without installing or importing PyTorch, Transformers, `mlx-lm`, or the upstream runtime.

## Broader Intent

This change advances `mlx-speech` as a clean multi-family speech library rather than a collection of upstream wrappers. The public API, checkpoint format, dependency boundary, and Hugging Face release must remain maintainable after the initial dots.tts port lands.

## Approved Approach and Evidence

The approved checkpoint scope is **SOAR + MeanFlow**:

- SOAR is the quality-focused, self-corrective-aligned checkpoint recommended upstream for voice cloning.
- MeanFlow is the latency-focused distilled checkpoint and exercises a distinct four-evaluation solver without runtime classifier-free guidance.
- The official PyTorch reference is pinned at `.references/dots.tts` commit `5ed719e3d36f5a3f6d8037ca9a7009d4fd0520ba` (`v0.2.1`).
- The community MLX inference reference is pinned at `.references/dots-tts-mlx` commit `f64479f51a2a9d7093533732cae86e765d8fb96e` (`v0.7.0`).
- Official SOAR weights are pinned to Hugging Face revision `e3520f75254d0020a0406db31c51a79d00d22d55`; official MeanFlow weights are pinned to `25c53fb462e57087e52237daa5ea30df1c5cc328`.
- The official core checkpoint is BF16; the vocoder and CAM++ speaker checkpoint are FP32. The existing MLX port demonstrates end-to-end feasibility and a conservative int8 policy, but it does not establish compatibility with this repository's API, dependency policy, checkpoint contract, or quality gates.

Assumption for product review: publish one family repository, `appautomaton/dots-tts-mlx`, with separate SOAR/MeanFlow and BF16/int8 runtime subdirectories. A different repository split may be selected during planning only if it preserves the same four independently loadable artifacts and one authoritative model-card story.

## Required Outcome

### REQ-001 — Reproducible source acquisition and local layout

- Preserve the official Hugging Face snapshots unchanged under:
  - `models/dots_tts/soar/original/`
  - `models/dots_tts/mf/original/`
- Provide a reproducible, revision-pinned acquisition command or script.
- Keep original and converted weights gitignored; no model tensor enters Git history.
- Produce converted artifacts under each variant's `mlx-bf16/` and `mlx-int8/` directories.

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
  - `dots-tts-soar` → int8 default
  - `dots-tts-soar-bf16`
  - `dots-tts-mf` → int8 default
  - `dots-tts-mf-bf16`
- `generate(text, reference_audio=..., reference_text=...)` must support continuation voice cloning.
- `generate(text, reference_audio=...)` must support speaker-embedding-only cloning.
- The no-reference path may remain callable for upstream parity, but documentation must state that random-voice generation is not a quality-supported use of the released multi-speaker checkpoints.
- Model-specific controls must remain backend kwargs rather than expanding the shared protocol: explicit language code, solver steps, guidance scale, speaker scale, seed, and maximum audio patches.

### REQ-004 — Explicit MLX checkpoint conversion

- Conversion must remain separate from runtime loading.
- Convert PyTorch-shaped tensors into an explicit MLX-native layout, including convolution layouts and vocoder weight-normalization folding, instead of performing pervasive source-layout repair during inference.
- Convert `latent_stats.pt` into a Torch-free runtime asset, preferably safetensors.
- Preserve all required tokenizer/configuration assets and record model family, checkpoint mode, source revision, dtype, quantization predicate, group size, and component precision in machine-readable metadata.
- Strict loading must reject missing, unexpected, shape-mismatched, or metadata-inconsistent weights.

### REQ-005 — BF16 and mixed int8 artifacts

- BF16 is the unquantized runtime parity baseline for every component after conversion.
- The int8 artifact uses affine 8-bit, group size 64, for the Qwen2.5 Linear/Embedding trunk at minimum; activations remain BF16.
- The semantic encoder, DiT/MeanFlow head, AudioVAE/BigVGAN, CAM++, EOS head, and small conditioning projections remain BF16 unless component-level parity and end-to-end quality gates prove an additional int8 predicate safe.
- The exact quantization predicate must be serialized with the artifact and reconstructed automatically by the loader.
- Int8 is the default alias only after it passes the BF16 comparison gates.

### REQ-006 — Reference parity and behavioral validation

- Use the pinned official PyTorch source as the behavioral oracle and the pinned community MLX port as an additional implementation comparison, never as an imported runtime dependency.
- Add focused parity fixtures/tests for schedule construction, Qwen hidden/KV behavior, speaker features, AudioVAE encode/decode, semantic feedback, DiT/MeanFlow patch generation, weight mapping, and EOS behavior.
- Add end-to-end integration coverage for SOAR and MeanFlow, BF16 and int8, producing finite, non-silent 48 kHz waveform output.
- Evaluate int8 against BF16 on a fixed multilingual voice-cloning corpus before release. Aggregate ASR WER may regress by no more than 1 absolute percentage point, and speaker-similarity cosine may regress by no more than 0.02. Any weaker result blocks the int8 default and publication claim.
- Do not describe quantization as lossless unless the recorded evaluation supports that statement.

### REQ-007 — Hugging Face release and model card

- Publish validated artifacts under the `appautomaton` organization using the repository's resumable upload workflow.
- The proposed family layout is:

  ```text
  appautomaton/dots-tts-mlx/
    README.md
    soar/mlx-bf16/
    soar/mlx-int8/
    mf/mlx-bf16/
    mf/mlx-int8/
  ```

- Original upstream checkpoints must not be included in the MLX release repository.
- Register explicit upload targets so publication can resume and cannot accidentally upload `original/` directories.
- The model card must include upstream sources and revisions, Apache-2.0 terms, architecture and supported languages, variant/precision matrix, exact quantization scope, memory and quality measurements, local/Hugging Face usage, known limitations, and voice-cloning consent/misuse guidance.
- Each published subdirectory must load by Hugging Face repo plus subdirectory resolution and complete an end-to-end waveform smoke test after upload.

### REQ-008 — Documentation and verification integration

- Add a dots.tts model-family guide covering checkpoint choice, cloning modes, generation controls, local layout, conversion, quantization, expected memory, and limitations.
- Update the model registry and Hugging Face release documentation without introducing dots-specific parameters into unrelated adapters.
- Add unit, checkpoint, runtime, and opt-in integration tests at the repository's established tiers.

## Acceptance Criteria

1. A revision-pinned acquisition workflow reconstructs both `original/` directories, and a four-way conversion matrix reconstructs SOAR/MF × BF16/int8 without modifying the originals.
2. All four converted directories are self-contained, strict-load successfully by local path, contain only Torch-free runtime assets, and record their source revision and precision policy.
3. Every converted tensor is either consumed by a named MLX module or explicitly rejected; conversion and load reports contain no unexplained missing, unexpected, or shape-mismatched keys.
4. Each of the four artifacts completes continuation cloning and speaker-only cloning and returns a finite, non-silent mono `mx.array` at 48,000 Hz through `mlx_speech.tts.load(...).generate(...)`.
5. Fixed-seed generation is repeatable for the same artifact and inputs, and SOAR and MeanFlow use their intended solver semantics and defaults.
6. Runtime dependency guards demonstrate that importing and generating with dots.tts does not import PyTorch, torchaudio, Transformers, `mlx-lm`, or either `.references/` checkout.
7. Component parity tests pass their checked-in numerical tolerances against pinned oracle fixtures; end-to-end BF16 output passes the fixed intelligibility and speaker-conditioning smoke corpus.
8. Int8 checkpoint bytes are at least 25% smaller than BF16 for each checkpoint, and the fixed corpus meets the WER and speaker-similarity regression limits in REQ-006.
9. The four aliases resolve to the intended artifact, with int8 as the default only after criterion 8 passes.
10. `pytest tests/unit/`, the relevant checkpoint/runtime tiers, and opt-in four-artifact waveform integration tests pass before publication.
11. The Hugging Face repository contains the four runtime subdirectories and authoritative model card, excludes original checkpoints, loads remotely by variant, and passes a post-upload waveform smoke test.

## Constraints and Risks

- dots.tts is a continuous autoregressive model; memory grows with prompt and generated length even when the Qwen trunk is quantized.
- The upstream and community runtimes contain optimization and caching choices that require independent parity proof before adoption.
- Flow-matching and vocoder components are precision-sensitive; blanket int8 quantization is prohibited without evidence.
- Reference audio processing, convolution layout, weight normalization, and latent scaling are high-risk parity boundaries.
- High-fidelity voice cloning creates impersonation and consent risks. Documentation and model cards must make authorized-use expectations and synthetic-audio disclosure explicit.
- Publishing to Hugging Face is an external state change authorized by this spec, but execution must stop at a human-action checkpoint if credentials or organization permissions are unavailable.

## Scope Coverage

### Included

- SOAR and MeanFlow inference.
- BF16 and mixed W8A-BF16 runtime artifacts.
- Continuation and speaker-only voice cloning with 48 kHz waveform output.
- Local conversion, quantization, parity, integration tests, aliases, documentation, Hugging Face packaging, model card, upload, and remote smoke validation.

### Deferred / Not in Scope

- `dots.tts-base`: deferred because SOAR supersedes it for the quality-focused released use case and the user selected SOAR + MeanFlow.
- MLX training, fine-tuning, SOAR alignment, or MeanFlow distillation: the MLX target is inference-only; official PyTorch remains the training reference.
- A new public streaming TTS protocol: the current unified interface is non-streaming; internal incremental state may be implemented where needed, while public streaming requires a separate cross-family API decision.
- Persistent enrolled-speaker profiles and long-text sentence stitching: useful follow-on runtime features, not required for first family parity.
- Swift/iOS runtime support.
- Dependency-heavy automatic text normalization or language detection; explicit language codes and already-normalized text are sufficient for this change.

## Anti-Goals

- Do not vendor, import, or package code from `.references/` as the runtime.
- Do not label a Torch-backed, Transformers-backed, or `mlx-lm`-backed path as MLX support.
- Do not stop at latent/token generation; end-to-end means waveform output.
- Do not mutate or republish official source checkpoints as if they were MLX artifacts.
- Do not commit model weights, reference audio, or generated evaluation audio to Git.
- Do not broaden the shared TTS protocol with dots-specific controls.
- Do not claim upstream benchmark parity, lossless quantization, or real-time performance without reproduced evidence.
