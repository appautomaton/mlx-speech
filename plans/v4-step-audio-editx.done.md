# v4: Step-Audio-EditX — MLX-Native Full Waveform Inference

**Status: ACTIVE** (2026-03-31)

## Summary

Bring Step-Audio-EditX into mlx-speech as a new inference-first family:
prompt audio + text in, waveform out, with no torch-backed runtime under the
MLX label.

The implementation must be grounded in the shipped inference path under:

- `models/stepfun/step_audio_editx/original/`
- `models/stepfun/step_audio_tokenizer/original/`
- `.references/Step-Audio-EditX/`

The Step1 checkpoint is a decoder-only LM with no rotary embeddings and a
sqrt-ALiBi causal attention bias. We do not infer alternate model structure
from unrelated training configs or older wrappers.

## Progress Snapshot

Landed in the first implementation slice:

- active v4 repo plan and AGENTS registration
- Step1 config parsing, decoder model, grouped KV cache, and checkpoint loading
- Step-Audio text tokenizer and prompt packing via converted `tokenizer.json`
- dedicated `step_audio_tokenizer` family for asset loading and exact
  interleave/offset/prompt-token arithmetic
- tokenizer processor layer for waveform preprocessing, vq02 codebook
  assignment, and vq06 chunk/log-mel preparation
- torch-free loading of the local FunASR `model.pt` archive into MLX arrays
- pure-Python ONNX graph loading for the local semantic tokenizer checkpoint
- MLX runtime for the full `vq02` path:
  waveform -> fbank/LFR/CMVN -> SANM encoder -> clustered linguistic token ids
- MLX runtime for the full `vq06` path:
  waveform/log-mel -> semantic encoder -> nearest-codebook token ids
- manual audit scripts for Step1 checkpoint alignment and tokenizer assets
- focused tests for Step1 core, prompt tokenizer behavior, and tokenizer
  packing/asset loading
- focused tests for `vq02` and `vq06` checkpoint alignment and runtime behavior

Still remaining for end-to-end waveform support:

- high-level generation wrapper and CLI
- MLX-native conversion flow for full family artifacts

## Source Truth

Primary upstream runtime truth:

- `.references/Step-Audio-EditX/modeling_step1.py`
- `.references/Step-Audio-EditX/tokenizer.py`
- `.references/Step-Audio-EditX/tts.py`
- `.references/Step-Audio-EditX/config/prompts.py`
- `.references/Step-Audio-EditX/stepvocoder/cosyvoice2/...`

Checkpoint truth:

- `models/stepfun/step_audio_editx/original/config.json`
- `models/stepfun/step_audio_editx/original/model.safetensors.index.json`
- `models/stepfun/step_audio_editx/original/tokenizer.model`
- `models/stepfun/step_audio_editx/original/tokenizer_config.json`
- `models/stepfun/step_audio_editx/original/CosyVoice-300M-25Hz/*`
- `models/stepfun/step_audio_tokenizer/original/*`

MLX references:

- `.references/mlx/` for operator behavior, conv/cache semantics, quantization,
  and lazy-eval patterns only

Dependency and integration rules for this family:

- no new Python packages without explicit user approval
- prefer existing repo dependencies: `mlx`, `numpy`, `safetensors`, `tokenizers`
- do not introduce torch, sentencepiece, ONNX Runtime, or convenience-only runtime deps
- if a slice is blocked by a missing dependency, stop and report the exact blocker
- fit new code into the existing `mlx-speech` package, loader, and test patterns

## Final Runtime Shape

```text
prompt wav
  -> Step-Audio dual tokenizer
       -> vq02 linguistic codes
       -> vq06 semantic codes
  -> prompt mel features
  -> speaker embedding
  -> Step1 decoder LM generates mixed audio tokens
  -> CosyVoice flow predicts mel for generated portion
  -> HiFT decodes waveform
```

Important invariants:

- do not replace the real waveform path with a token-only path
- do not collapse tokenizer, LM, and waveform decoder into one opaque module
- do not assume prompt mel or speaker embedding are optional in the waveform path
- do not assume the integrated `step_audio.py` path is the primary inference truth

## Public API

Add a dedicated model family plus a high-level wrapper.

Planned structure:

- `src/mlx_speech/models/step_audio_tokenizer/`
- `src/mlx_speech/models/step_audio_editx/`
- `src/mlx_speech/generation/step_audio_editx.py`

Public wrapper:

- `StepAudioEditXModel.from_dir(...)`
- `StepAudioEditXModel.from_path(...)`
- `clone(prompt_audio, prompt_sample_rate, prompt_text, target_text)`
- `edit(prompt_audio, prompt_sample_rate, prompt_text, edit_type, edit_info=None, target_text=None)`

The wrapper takes in-memory arrays and explicit sample rates. File I/O belongs
in the CLI only.

Return one synthesis result type with:

- `waveform`
- `sample_rate`
- `generated_token_ids`
- `stop_reached`
- additional status metadata when needed for debugging

Keep tokenizer and waveform internals importable through the model packages,
but do not widen the top-level public API beyond the wrapper.

## Implementation Stages

### Stage 1 — Plan and repo state

- create this active plan
- register `v4` in `AGENTS.md`
- keep the scope explicit: full waveform support, not token-only bring-up

### Stage 2 — Step1 config, model, and checkpoint loading

Implement the shipped Step1 checkpoint exactly:

- 32 pre-norm decoder blocks
- `hidden_size=3072`
- `intermediate_size=8192`
- `num_attention_heads=48`
- `num_attention_groups=4`
- `head_dim=64`
- untied `embed_tokens` and `lm_head`
- RMSNorm with fp32 accumulation
- SwiGLU MLP
- final RMSNorm
- no dropout
- no q/k/v/o or MLP biases

Attention and cache rules:

- grouped-query attention, not full K/V heads
- no RoPE, no learned position embeddings
- explicit sqrt-ALiBi causal bias from the shipped formula
- no blind reuse of HF `DynamicCache`
- define a family-local grouped KV cache after K/V reshape
- reproduce the transparent fallback math, not the opaque `torch.ops.Optimus` path

Checkpoint rules:

- support original local checkpoints first
- validate exact key/shape alignment against the index and state dict
- then support MLX-native checkpoints and `mlx-int8`

### Stage 3 — Text tokenizer and prompt builder

Support the shipped text tokenizer assets:

- converted `tokenizer.json`
- `tokenizer_config.json`
- shipped chat template behavior

Use `tokenizers` at runtime. Generate `tokenizer.json` from the shipped
SentencePiece model as a conversion/prep step instead of carrying
`sentencepiece` in the runtime dependency set.

Prompt rules:

- clone mode uses the shipped clone system template
- preserve `[speaker_start] ... [speaker_end]`
- preserve upstream `prompt_speaker = "debug"` behavior
- edit mode uses the shipped audio-edit system prompt
- user message is instruction text + newline + audio token string

### Stage 4 — Step-Audio tokenizer family

Implement a separate reusable tokenizer family:

Verified landed portion:

- `src/mlx_speech/models/step_audio_tokenizer/{config,checkpoint,packing}.py`
- `src/mlx_speech/models/step_audio_tokenizer/processor.py`
- `src/mlx_speech/models/step_audio_tokenizer/vq02.py`
- `src/mlx_speech/models/step_audio_tokenizer/vq06.py`
- `src/mlx_speech/checkpoints/pytorch_pickle.py`
- `src/mlx_speech/checkpoints/onnx_proto.py`
- `tests/test_step_audio_tokenizer.py`
- `tests/test_step_audio_tokenizer_processor.py`
- `tests/test_step_audio_editx_tokenizer.py`
- `tests/test_step_audio_tokenizer_vq02.py`
- `tests/test_step_audio_tokenizer_vq06.py`
- exact 2+3 interleave/deinterleave helpers
- exact raw/prompt/mixed token offset helpers
- `<audio_n>` packing/parsing helpers
- local tokenizer asset loading and codebook validation
- local FunASR `model.pt` loading and tensor-shape validation
- vq02 waveform preprocessing and codebook clustering helpers
- vq06 chunk planning and Whisper-style 128-mel front-end preparation
- pure-MLX `vq02` execution:
  - local FunASR config parsing
  - local `model.pt` checkpoint remapping
  - minimal SANM encoder inference path
  - waveform -> linguistic token ids runtime
- pure-MLX `vq06` execution:
  - local ONNX graph parsing without runtime deps
  - semantic encoder reconstruction from local graph/source evidence
  - quantizer/codebook bring-up
  - waveform -> semantic token ids runtime

Stage 4 status:

- [x] asset loading, codebook validation, and exact token packing helpers
- [x] waveform preprocessing, vq02 codebook clustering, and vq06 chunk/log-mel preparation
- [x] vq02 FunASR config/model bring-up from the vendored `funasr_detach` sources
- [x] vq02 checkpoint/loading strategy for the Paraformer streaming encoder path
- [x] local architecture/source truth recovery for `speech_tokenizer_v1.onnx`
- [x] MLX execution path for the semantic tokenizer network

Stage 4 is complete. The remaining plan work starts at Stage 5.

Sequence invariants:

- `vq02` and `vq06` remain separate internally
- mixed interleave is fixed `2 + 3`
- prompt string form uses `<audio_n>`
- raw/mixed/vocoder offsets match upstream arithmetic exactly
- dual-codebook reshape matches CosyVoice `_reshape(...)`

### Stage 5 — CosyVoice waveform path

Implement the non-stream waveform path first.

Required pieces:

- frontend mel extraction
- speaker embedding extraction
- dual-codebook embedding
- flow inference that consumes:
  - generated token tensor
  - prompt token tensor
  - prompt mel features
  - speaker embedding
- prompt mel interpolation to `prompt_token_len * 2`
- HiFT waveform decode

Restrictions:

- do not assume BigVGAN; shipped config is HiFT
- do not assume bundled `speech_tokenizer_v1.onnx` is part of waveform decode
- do not expose streaming in the first public runtime surface

Verified landed first slice:

- `src/mlx_speech/models/step_audio_editx/frontend.py`
- `src/mlx_speech/models/step_audio_editx/campplus.py`
- `src/mlx_speech/models/step_audio_editx/flow.py`
- `src/mlx_speech/models/step_audio_editx/flow_model.py`
- `src/mlx_speech/models/step_audio_editx/hift.py`
- `tests/test_step_audio_editx_frontend.py`
- `tests/test_step_audio_editx_campplus.py`
- `tests/test_step_audio_editx_flow.py`
- `tests/test_step_audio_editx_hift.py`
- `tests/test_step_audio_editx_nonstream_integration.py`
- local `cosyvoice.yaml` mel-config parsing
- MLX/NumPy prompt-mel extraction for `token2wav_nonstream`
- mel filterbank now honors the shipped `fmin` / `fmax` bounds
- frontend mel behavior test covers the `fmax=8000` cutoff used by CosyVoice
- pure-MLX CAMPPlus checkpoint loading and speaker-embedding inference from the
  local `campplus.onnx` export
- frontend `extract_spk_embedding(...)` now routes through the MLX CAMPPlus
  runtime instead of ONNX Runtime
- checkpoint-backed non-stream flow conditioning helpers:
  - exact shipped `_reshape(...)` arithmetic for mixed prompt tokens
  - nearest-neighbor prompt-mel interpolation to `prompt_token_len * 2`
  - dual-codebook embedding from `flow.pt`
  - normalized speaker projection from `flow.pt`
  - preparation of concatenated prompt/generated token embeddings for the flow encoder
- pure-MLX non-stream flow runtime from the shipped `flow.pt` checkpoint:
  - `UpsampleConformerEncoderV2` encoder path
  - `encoder_proj`
  - `CausalConditionalCFM` non-stream Euler solver
  - `DiT` estimator path
  - real local-asset mel inference smoke path through prompt mel, prompt/generated tokens, and speaker embedding
- pure-MLX HiFT vocoder runtime from the shipped `hift.pt` checkpoint:
  - F0 predictor
  - source generation
  - HiFT generator decode path
  - finite waveform output from synthetic mel input
- first local non-stream waveform smoke path:
  - prompt audio tokenization (`vq02` + `vq06`)
  - Step1 prompt building and short greedy audio-token generation
  - prompt mel + speaker embedding
  - flow conditioning + flow mel inference
  - HiFT waveform decode
  - manual local integration test gated behind `RUN_LOCAL_INTEGRATION=1`

Remaining Stage 5 work:

- none

Current Stage 5 queue:

- [x] port the CosyVoice frontend mel extraction path used by `token2wav_nonstream`
- [x] port speaker embedding extraction without ONNX Runtime in the MLX runtime
- [x] port the non-stream prompt-conditioning path used by `token2wav_nonstream`
- [x] port the non-stream flow encoder/decoder path
- [x] port HiFT waveform decode
- [x] validate prompt token reshape/interpolation against the shipped CosyVoice path

Stage 5 is complete. The remaining work begins at Stage 6.

### Stage 6 — Wrapper, CLI, and conversion

Verified landed Stage 6 slice:

- `src/mlx_speech/generation/step_audio_editx.py`
- `src/mlx_speech/generation/__init__.py`
- `scripts/generate_step_audio_editx.py`
- `scripts/convert_step_audio_tokenizer.py`
- `scripts/convert_step_audio_editx.py`
- `tests/test_step_audio_editx_generation.py`
- `tests/test_step_audio_editx_public_integration.py`
- `tests/test_generate_step_audio_editx_script.py`

What landed:

- public `StepAudioEditXModel` wrapper with:
  - `from_dir(...)`
  - `from_path(...)`
  - `clone(...)`
  - `edit(...)`
- wrapper orchestration over the landed runtime pieces:
  - Step1 LM
  - Step-Audio tokenizer family (`vq02` + `vq06`)
  - CosyVoice frontend
  - CAMPPlus speaker embedding
  - non-stream flow conditioning
  - non-stream flow model
  - HiFT waveform decode
- local file-based CLI for clone/edit generation
- tokenizer packaging script for reusable local runtime assets
- Step1 original-to-`mlx-int8` conversion script with runtime-asset copying
- gated public-API local clone smoke test

Verified validation:

- focused wrapper + CLI tests:
  - `6 passed, 1 skipped`
- gated local public wrapper smoke:
  - `1 passed`
- tokenizer packaging script:
  - packaged to `/tmp/step_audio_tokenizer_pkg`
  - `vq02_alignment_exact: True`
  - `vq06_alignment_exact: True`
- Step1 conversion script:
  - converted to `/tmp/step_audio_editx_mlx_int8`
- full repo:
  - `227 passed, 8 skipped`
- full lint:
  - `uv run ruff check .` passed

Runtime target order:

- original checkpoint path first
- MLX-native saved checkpoints next
- `mlx-int8` as the normal default only after parity is trustworthy

Remaining Stage 6 work:

- none

Stage 6 is complete. The remaining work begins at Stage 7.

### Stage 7 — Docs and publication readiness

Verified landed Stage 7 slice:

- `docs/step-audio-editx.md`
- `docs/README.md`
- `README.md`
- `docs/references.md`
- `scripts/README.md`

What landed:

- a dedicated Step-Audio model-family guide
- README model-family entry and local usage examples
- explicit original-first runtime guidance for the public wrapper and CLI
- documentation for the gated local public integration path
- `docs/references.md` remains pinned to the actual local Step-Audio upstream commit
- the wrapper export through `src/mlx_speech/generation/__init__.py` is already landed and no longer tracked as open work

Publication-readiness truth:

- end-to-end local waveform generation is working through MLX
- the public wrapper and CLI are landed
- the runtime is still documented as original-first by default
- gated local integration remains manual on purpose

Remaining Stage 7 work:

- none

Stage 7 is complete. The v4 implementation scope is complete.

## Validation

Focused unit tests:

- Step1 config parsing from shipped `config.json`
- Step1 checkpoint alignment against original weights
- grouped-KV cache growth and shape behavior
- sqrt-ALiBi bias exactness
- RMSNorm fp32 accumulation behavior
- tokenizer interleave/deinterleave and offset math
- `<audio_n>` prompt packing
- CosyVoice dual-codebook reshape equivalence
- prompt mel interpolation shape contract
- flow and HiFT output-shape tests

Focused local integration tests:

- mark as `local_integration`
- manual only
- one clone smoke case
- one edit smoke case
- original checkpoint path
- `mlx-int8` path

Manual audit scripts:

- Step1 checkpoint/key audit
- tokenizer asset/config audit
- end-to-end local smoke generation

Do not add a default autorun integration suite for this family.

## Assumptions and defaults

- final scope is full waveform inference
- public surface is wrapper + internals
- runtime target is original first, then MLX-native and `mlx-int8`
- runtime uses `tokenizers` and a converted `tokenizer.json`
- `.references/mlx` is a behavior reference, not an architecture source
- no overlapping GPU bring-up runs while this family is being implemented
