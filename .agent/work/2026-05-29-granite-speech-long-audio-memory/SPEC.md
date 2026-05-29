# Granite Speech Long-Audio Memory And Benchmark Spec

## Bounded Goal

Fix Granite Speech ASR runtime memory behavior, add coarse MLX memory telemetry, and re-run `/tmp`-only long-audio transcription checks against public-domain audio with known scripts to measure efficiency and accuracy after the fix.

## Broader Intent

The user wants IBM Granite Speech ASR to behave like the repo's other local MLX inference paths: predictable local-path execution, bounded memory growth, no heavyweight reference-runtime dependencies, and clear diagnostic evidence for long-form audio behavior rather than anecdotal Activity Monitor observations.

## Work Scale And Shape

- Scale: medium runtime hardening and diagnostic benchmark change.
- Shape: runtime refactor plus benchmark/diagnostic instrumentation.

## Selected Lenses

- Product: determine whether the local Granite Speech path is practically usable for long audio and what its limits are.
- Engineering: remove pathological memory allocation and align implementation style with existing MLX runtimes.
- Runtime: track memory at meaningful lifecycle boundaries without constant polling.

## Source Evidence

- `src/mlx_speech/models/granite_speech_asr/language_model.py` currently hand-materializes fp32 attention scores and weights with `[B, heads, L, L]` shape, and manually repeats GQA KV heads before attention.
- `src/mlx_speech/models/granite_speech_asr/language_model.py` currently allocates `GraniteKVCache` with default `mx.float32`.
- `src/mlx_speech/generation/granite_speech_asr.py` currently validates context after feature extraction and encoder/projector execution.
- `/tmp` exploratory run found LibriVox / Project Gutenberg chapter `Tracked by a Catamount` at about 17:49 with a matching public-domain text script; whole-file prompt preflight is about 10,710 prompt tokens and exceeds the local model's 4,096-token context.
- Chunked `/tmp` exploratory run completed mechanically but exposed uneven transcript coverage and very high memory pressure.

## Required Outcome

### Runtime Behavior

- Granite LM attention must use MLX's efficient attention primitive for prefill/decode instead of explicit `scores = q @ k.T`, explicit `softmax(scores)`, and explicit GQA KV tiling.
- KV cache allocation must use an appropriate runtime/model dtype instead of unconditional fp32, and cache length accounting must remain exact across prefill and decode.
- Context-window validation for audio prompts must happen before encoder/projector work for file or array audio when the preflight shape can determine that the request cannot fit.
- Generation must not retain per-step logits, hidden-state history, large audio feature arrays, or stale chunk data beyond what is needed for the returned result.
- Existing Granite Speech public behavior and checkpoint loading must remain compatible with the completed ASR implementation.

### Memory Telemetry

- Add explicit, coarse memory metrics using MLX memory APIs at lifecycle boundaries such as before model load, after model load, before transcription, after prefill, after decode, and after cleanup.
- Telemetry must use snapshots only; do not add background polling loops, per-token polling, timers that spin, or external process monitors.
- Summary artifacts for diagnostic scripts must include memory fields where available: active memory, cache memory, peak memory, and a reset/clear-cache boundary where appropriate.

### Long-Audio Checking

- Use `/tmp` for downloaded audio, extracted scripts, chunks, transcripts, and summaries during exploratory checks unless the user explicitly asks to persist artifacts in the repo.
- Use public-domain audio with matching public-domain text scripts, with LibriVox / Project Gutenberg as the default source family.
- Re-run a greater-than-10-minute check after the runtime memory fix, using context-safe chunking when the full audio would exceed the model context.
- Report efficiency and behavior with duration, chunk count, prompt tokens, generated tokens, wall time, RTF / RTFx, peak memory, non-empty status, and an accuracy/coverage comparison against the known script.

## Constraints And Risks

- Keep the runtime pure MLX: no `torch`, `torchaudio`, `transformers`, `mlx_lm`, `mlx_audio`, vLLM, or ONNX runtime dependencies under `src/mlx_speech`.
- Do not claim single-prompt support for 10+ minute files when the model context does not support it; chunking is the expected long-form path.
- Do not write benchmark media, transcripts, or summaries under repo `outputs/` during this investigation unless explicitly requested.
- Avoid accuracy gates that depend on exact punctuation or casing; use meaningful text-normalized comparison.
- The efficient attention change is numerically sensitive. Tests must protect shape, cache, and small-model equivalence behavior before long-audio benchmarking.

## Acceptance Criteria

- Granite LM attention no longer explicitly materializes `[B, heads, L, L]` scores/weights in the local implementation, and uses `mx.fast.scaled_dot_product_attention` or an equivalently efficient MLX primitive that supports GQA without pre-tiling KV heads.
- Unit tests prove tiny Granite LM prefill and cached decode still work, cache length increments correctly, and cache dtype follows the configured/runtime dtype.
- A preflight/context test proves an over-context long audio request fails before `get_audio_features(...)` or encoder/projector execution.
- The default unit suite remains weight-free. Local Granite verification still passes where weights are present: strict checkpoint full-load and runtime smoke must stay skip-gated for users without IBM checkpoint files.
- Diagnostic scripts can emit coarse memory metrics in their summary without continuous polling.
- A manually invoked `/tmp` long-audio run using a public-domain >10-minute file and matching script completes through the fixed path with context-safe chunking, records memory and timing metrics, and produces an accuracy/coverage report against the script.
- The final report includes before/after interpretation of memory behavior and efficiency, including whether the prior 100+ GB memory-pressure pattern is resolved or still reproducible.

## Anti-Goals

- Do not add a dependency on IBM's Transformers path, `mlx_lm`, `mlx_audio`, torch, vLLM, or external ASR services.
- Do not optimize by disabling KV cache entirely unless a measured fallback is explicitly marked as diagnostic-only.
- Do not introduce constant memory polling or background monitoring.
- Do not commit downloaded audio files, generated chunk WAVs, transcripts, benchmark summaries, or Gutenberg/LibriVox text dumps.
- Do not broaden into diarization, translation quality, punctuation restoration, or a full production long-form transcript editor.

## Assumptions

- The current model context is fixed at the local Granite config's `max_position_embeddings`; long audio must be chunked unless a future model/config changes that.
- LibriVox / Project Gutenberg sources are acceptable for temporary `/tmp` benchmarking because they provide public-domain audio plus text.
- Activity Monitor pressure is useful as a symptom, but acceptance evidence should come from MLX memory snapshots and reproducible commands.
