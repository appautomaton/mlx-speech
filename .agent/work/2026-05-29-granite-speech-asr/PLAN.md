# Granite Speech ASR Plan

## Goal

Implement the approved Granite Speech ASR spec in `.agent/work/2026-05-29-granite-speech-asr/SPEC.md`.

## Architecture Approach

Use the design in `.agent/work/2026-05-29-granite-speech-asr/DESIGN.md`: a local `granite_speech_asr` model family with pure-MLX feature extraction, Conformer encoder, QFormer projector, Granite causal LM, checkpoint loading, bounded memory behavior, and greedy generation. Do not route runtime through `mlx-audio`, `mlx_lm`, `transformers`, torch, vLLM, or ONNX.

## Execution Routing And Topology

- Default execution: direct.
- Continue through all slices after each slice verifies.
- Parallel-safe groups: none. Later slices depend on earlier module contracts and checkpoint key names.
- Checkpoints: none. Human review is useful before execution because the LM/checkpoint surface is large, but no slice requires a human decision mid-plan.

## Ordered Slice Sequence

### Slice 1: Family Contracts And Tokenizer

**Objective:** Add Granite Speech ASR package scaffolding, config parsing, tokenizer/prompt rendering, and local-family dispatch for `model_type: granite_speech`.

**Acceptance criteria:**

- `GraniteSpeechConfig.from_path(...)` parses the downloaded upstream `config.json` and preserves encoder, projector, text, audio token, and dtype fields.
- Tokenizer wrapper loads `tokenizer.json`, `added_tokens.json`, and `chat_template.jinja`; `<|audio|>` resolves to id `100352`.
- Prompt rendering matches the checked-in chat template shape: `USER: <audio tokens>prompt\n ASSISTANT:`.
- ASR family resolution returns Granite for a local directory with `model_type: granite_speech`.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_config.py tests/unit/test_granite_speech_tokenizer.py tests/unit/test_asr_registry.py`

**Touches:** `src/mlx_speech/models/granite_speech_asr/`, `src/mlx_speech/asr/_registry.py`, unit tests.

**Status:** complete
**Evidence:** added Granite config/tokenizer package scaffolding and registry detection; `.venv/bin/python -m pytest tests/unit/test_granite_speech_config.py tests/unit/test_granite_speech_tokenizer.py tests/unit/test_asr_registry.py` passed (8 tests); `.venv/bin/python -m pytest tests/unit/` passed (325 tests).
**Risks / next:** none for slice 1; continue to feature extraction and audio token accounting.

### Slice 2: Feature Extraction And Audio Token Accounting

**Objective:** Implement Granite Speech's local audio frontend and token-count calculation.

**Acceptance criteria:**

- Feature extraction accepts 16 kHz mono waveform arrays and returns `[1, frames, 160]` pair-stacked encoder input.
- STFT/mel settings match the model metadata: `n_fft=512`, `win_length=400`, `hop_length=160`, `n_mels=80`, HTK mel.
- Odd mel-frame counts are trimmed before pair stacking.
- Audio token count follows `ceil(encoder_frames / window_size) * (window_size // downsample_rate)`.
- Feature extraction exposes a preflight helper that computes encoder frames and projected audio tokens from sample count before allocating prompt embeddings.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_feature_extraction.py`

**Touches:** `src/mlx_speech/models/granite_speech_asr/feature_extraction.py`, processor tests.

**Status:** complete
**Evidence:** added pure-numpy Granite HTK log-mel frontend, pair-stacking, preflight sizing, and audio-token count tests; `.venv/bin/python -m pytest tests/unit/test_granite_speech_feature_extraction.py` passed (6 tests); `.venv/bin/python -m pytest tests/unit/` passed (331 tests).
**Risks / next:** feature numerics are source-shaped but not reference-parity checked; continue to checkpoint sanitizer and key accounting.

### Slice 3: Checkpoint Loader And Sanitizer

**Objective:** Add deterministic loading and key accounting for original Granite Speech `.safetensors` shards.

**Acceptance criteria:**

- Loader supports sharded `model.safetensors.index.json` through existing checkpoint utilities.
- Sanitizer drops `num_batches_tracked` keys and transposes original 1D conv weights for `up_conv`, `down_conv`, and `depth_conv`.
- Alignment report distinguishes checkpoint-only, model-only, and shape mismatch keys.
- Real local checkpoint header test confirms 954 indexed keys and component namespaces when files are present; it skips cleanly when absent.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_checkpoint_unit.py tests/checkpoint/test_granite_speech_checkpoint.py`

**Touches:** `src/mlx_speech/models/granite_speech_asr/checkpoint.py`, checkpoint tests.

**Status:** complete
**Evidence:** added Granite sharded checkpoint loader, sanitizer, Conv1d layout fix tracking, alignment report, and index-only local checkpoint inspection; corrected pytest unit/checkpoint basename collision in the slice command; `.venv/bin/python -m pytest tests/unit/test_granite_speech_checkpoint_unit.py tests/checkpoint/test_granite_speech_checkpoint.py` passed (6 tests); `.venv/bin/python -m pytest tests/unit/` passed (336 tests).
**Risks / next:** strict full-model alignment waits for model module paths in later slices; continue to Conformer encoder.

### Slice 4: Conformer Encoder

**Objective:** Implement the Granite Speech Conformer CTC encoder in MLX with tiny-config forward tests.

**Acceptance criteria:**

- Encoder contains input projection, 16 configurable Conformer blocks, block attention with relative position embeddings, convolution module with BatchNorm-style running stats, and midpoint CTC self-conditioning.
- Tiny encoder forward preserves `[B, T, hidden]` shape and handles non-multiple-of-context lengths.
- Module parameter names align with checkpoint keys under `encoder.*`.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_encoder.py`

**Touches:** `src/mlx_speech/models/granite_speech_asr/encoder.py`, unit tests.

**Status:** complete
**Evidence:** added MLX Conformer encoder with block attention, relative position embeddings, Conv1d module with BatchNorm-style running stats, midpoint CTC self-conditioning, and checkpoint-shaped parameter names; `.venv/bin/python -m pytest tests/unit/test_granite_speech_encoder.py` passed (4 tests); `.venv/bin/python -m pytest tests/unit/` passed (340 tests).
**Risks / next:** numerical parity against reference is still deferred per engineering review; continue to QFormer projector and audio embedding replacement.

### Slice 5: QFormer Projector And Audio Embedding Replacement

**Objective:** Implement the QFormer projector and text/audio embedding merge boundary.

**Acceptance criteria:**

- Projector pads encoder states into 15-frame windows, applies 3 learned queries per window, and projects to Granite hidden size 2048.
- Tiny projector forward returns `[B, audio_tokens, text_hidden]`.
- Audio embedding replacement substitutes projected audio features at `<|audio|>` positions and leaves non-audio token embeddings intact.
- Mismatched audio-token counts raise a clear error instead of silently truncating.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_projector.py tests/unit/test_granite_speech_processor.py`

**Touches:** `src/mlx_speech/models/granite_speech_asr/projector.py`, `processor.py`, unit tests.

**Status:** complete
**Evidence:** added checkpoint-shaped QFormer projector with 15-frame window padding and 3 queries per window, plus strict audio embedding replacement helpers; `.venv/bin/python -m pytest tests/unit/test_granite_speech_projector.py tests/unit/test_granite_speech_processor.py` passed (6 tests); `.venv/bin/python -m pytest tests/unit/` passed (346 tests).
**Risks / next:** projector numerical parity is not yet checked against reference weights; continue to local Granite causal LM primitives.

### Slice 6: Granite Causal LM And Greedy Decode Primitives

**Objective:** Implement the local Granite LM pieces needed for ASR prefill and token-by-token greedy decoding.

**Acceptance criteria:**

- LM supports this checkpoint's hidden size, 40 configurable layers, GQA, RoPE, SwiGLU MLP, RMSNorm, untied `lm_head`, KV cache, and causal mask.
- Forward applies `embedding_multiplier`, `attention_multiplier`, `residual_multiplier`, and `logits_scaling`.
- Tiny LM tests cover prefill with input embeddings and one-step cached decode.
- No `mlx_lm` imports exist under `src/mlx_speech`.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_language_model.py && ! rg -n "mlx_lm" src/mlx_speech`

**Touches:** `src/mlx_speech/models/granite_speech_asr/language_model.py`, model tests.

**Status:** complete
**Evidence:** added local Granite causal LM with RoPE, GQA, SwiGLU MLP, multiplier handling, KV cache prefill/decode helpers, and greedy next-token selection; `.venv/bin/python -m pytest tests/unit/test_granite_speech_language_model.py` passed (5 tests); `rg -n "mlx_lm" src/mlx_speech` found no matches; `.venv/bin/python -m pytest tests/unit/` passed (351 tests).
**Risks / next:** LM numerical parity against mlx-lm reference remains deferred per engineering review; continue to full model assembly and strict checkpoint loading.

### Slice 7: Full Model Assembly And Strict Load

**Objective:** Compose encoder, projector, and LM into a loadable Granite Speech model with strict checkpoint accounting.

**Acceptance criteria:**

- `GraniteSpeechModel(config)` exposes module paths matching sanitized checkpoint keys.
- Full-model alignment against the downloaded checkpoint succeeds except for intentional skipped keys.
- `from_dir(...)` loads config, tokenizer, processor, model weights, sets eval mode, and evaluates parameters.
- `from_dir(...)` does not retain a full checkpoint state dict on the runtime object after weights are loaded.
- Checkpoint test skips cleanly when local Granite files are absent.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_model.py tests/checkpoint/test_granite_speech_full_load.py`

**Touches:** `src/mlx_speech/models/granite_speech_asr/model.py`, `checkpoint.py`, `__init__.py`, checkpoint tests.

### Slice 8: Runtime Generation And ASR Adapter

**Objective:** Add the public Granite ASR runtime wrapper and adapter through existing ASR surfaces.

**Acceptance criteria:**

- `GraniteSpeechAsrModel.transcribe(...)` runs feature extraction, audio feature projection, prompt construction, mixed-embedding prefill, and greedy decode.
- Core generation returns structured transcription results only and does not write files or mutate `outputs/`.
- Generation validates `prompt_tokens + max_new_tokens <= config.text_config.max_position_embeddings` before prefill, and raises a clear `ValueError` when the request would exceed context.
- Generation retains KV cache and generated token ids only; it does not accumulate per-step logits or hidden-state history.
- `mlx_speech.asr.load(<local granite path>)` returns an adapter for `model_type: granite_speech`.
- Adapter accepts `str | Path | np.ndarray | mx.array` audio and returns `ASROutput`.
- Generic `mlx-speech asr --model <local granite path> --audio <file>` works without new CLI flags.

**Verification:** `.venv/bin/python -m pytest tests/unit/test_granite_speech_generation.py tests/unit/test_granite_speech_adapter.py tests/unit/test_granite_speech_memory_bounds.py`

**Touches:** `src/mlx_speech/generation/granite_speech_asr.py`, `src/mlx_speech/asr/__init__.py`, `src/mlx_speech/asr/_adapters/granite_speech.py`, tests.

### Slice 9: Gated Runtime Smoke, Diagnostics, And Documentation

**Objective:** Prove local end-to-end transcription on the downloaded checkpoint, add local transcript diagnostics under `outputs/granite_speech_asr/`, then document Granite Speech support.

**Acceptance criteria:**

- Runtime smoke test transcribes `models/ibm/granite_4_0_1b_speech/original/multilingual_sample.wav` into non-empty text when local files are present.
- `scripts/generate/granite_speech_asr.py` can transcribe one or more audio paths through the existing ASR generation pipeline and defaults output artifacts to `outputs/granite_speech_asr/`.
- The diagnostic script can use a curated local sample set from existing generated/reference WAVs under `outputs/smoke/generated/`, `outputs/dramabox/`, `outputs/clone_eval/manual/`, and `outputs/source/`.
- Diagnostic transcripts are written under `outputs/granite_speech_asr/transcripts/` with collision-safe paths; `summary.json` records input path, output path, non-empty status, and error text when a file fails.
- The diagnostic path never writes into `outputs/smoke/`, modifies existing sample WAV files, or makes exact transcript text a gate.
- Unit tests cover diagnostic output path construction and summary writing without loading real model weights.
- Test suite includes a dependency guard that fails if `src/mlx_speech` imports `torch`, `torchaudio`, `mlx_lm`, `transformers`, `vllm`, or `mlx_audio`.
- README/docs mention Granite Speech only after smoke support exists and describe local-path loading.
- `pytest tests/unit/` passes.

**Verification:** `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/test_granite_speech_full_load.py tests/runtime/test_granite_speech_smoke.py`

**Touches:** `scripts/generate/granite_speech_asr.py`, `tests/runtime/`, README/docs, dependency guard tests.

## Aggregate Verification Commands

| Slice | Command |
| --- | --- |
| 1 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_config.py tests/unit/test_granite_speech_tokenizer.py tests/unit/test_asr_registry.py` |
| 2 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_feature_extraction.py` |
| 3 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_checkpoint_unit.py tests/checkpoint/test_granite_speech_checkpoint.py` |
| 4 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_encoder.py` |
| 5 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_projector.py tests/unit/test_granite_speech_processor.py` |
| 6 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_language_model.py && ! rg -n "mlx_lm" src/mlx_speech` |
| 7 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_model.py tests/checkpoint/test_granite_speech_full_load.py` |
| 8 | `.venv/bin/python -m pytest tests/unit/test_granite_speech_generation.py tests/unit/test_granite_speech_adapter.py tests/unit/test_granite_speech_memory_bounds.py` |
| 9 | `.venv/bin/python -m pytest tests/unit/ tests/checkpoint/test_granite_speech_full_load.py tests/runtime/test_granite_speech_smoke.py` |

## Engineering Review Recommendation

Run `auto-eng-review` before execution. The plan touches a new local LM implementation, full-weight checkpoint alignment, and runtime dependency boundaries.

## Review: Engineering

- Verdict: approved_with_risks
- Strength: Purely additive family that mirrors the existing `cohere_asr` layout and `asr/` dispatch, with strictly sequential, independently testable slices, strong checkpoint-key accounting, and explicit import guards that enforce the no-`mlx_lm`/`transformers`/torch boundary.
- Concern: The two hand-ported numeric kernels (Conformer encoder, Granite LM) have no numerical parity check, so a subtle RoPE/multiplier/mel-scaling error would survive to the only end-to-end gate (Slice 9), which asserts non-empty text rather than correct text.
- Action: During Slice 9 (or as a dev-only check in Slice 4/6), add a numerical spot-check of encoder and LM forwards against the `.references` mlx_lm Granite reference, and assert the `multilingual_sample.wav` transcript matches expected content rather than merely being non-empty.
- Verified: data flow traced against reference pipeline; registry/adapter/cohere_asr layout and sharded loader confirmed in repo; reference's mlx_lm/transformers dependency confirmed as the boundary being replaced; checkpoint key counts reconciled (363+534+57=954); context-overflow and audio-token-mismatch error paths confirmed in ACs; empty-waveform shadow path noted as not enumerated.
