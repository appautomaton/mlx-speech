# Granite Speech ASR

## Bounded Goal

Add pure-MLX inference support for IBM Granite Speech 4.0 1B ASR in `mlx-speech`, loading local `.safetensors` checkpoints and producing text through the existing ASR interface.

## Broader Intent

Bring a newer open-weight ASR family into the library without weakening the project's runtime rules: no torch-backed inference, no `mlx_lm` runtime dependency, no `transformers` runtime dependency, and no framework-heavy abstraction layer.

## Work Scale And Shape

- Scale: capability-sized model-family port.
- Shape: parity-oriented inference implementation with checkpoint loading, feature extraction, tokenizer/prompt handling, model forward, greedy generation, adapter/CLI integration, and focused tests.
- Selected lenses: product, engineering, runtime.

## Target User

Apple Silicon users of `mlx-speech` who want a local open-weight ASR option beyond Cohere ASR, available through the same `mlx_speech.asr.load(...)` and `mlx-speech asr ...` surfaces.

## Source Evidence

- Upstream model metadata: `.references/granite-4.0-1b-speech/`
- Downloaded original weights: `models/ibm/granite_4_0_1b_speech/original/`
- MLX reference implementation: `.references/mlx-audio/mlx_audio/stt/models/granite_speech/`
- Existing ASR integration pattern: `src/mlx_speech/asr/`, `src/mlx_speech/models/cohere_asr/`, `src/mlx_speech/generation/cohere_asr.py`
- Existing local audio samples for diagnostic transcription: `outputs/smoke/generated/`, `outputs/dramabox/`, `outputs/clone_eval/manual/`, and `outputs/source/`

Verified local facts:

- Checkpoint revision: `bd87ab862416353633ea431fe49b1614003623c5`
- Original checkpoint layout: 3 safetensors shards, 954 keys, 4.3 GB on disk.
- Tensor namespaces: `language_model` 363 keys, `encoder` 534 keys, `projector` 57 keys.
- Dtypes: 938 BF16 tensors, 16 I64 tensors.
- `config.json` reports `model_type: granite_speech`, `audio_token_index: 100352`, `has_lora_adapter: false`, no quantization config.

## Required Outcome

The implementation must provide a coherent Granite Speech ASR family in `mlx-speech`:

- Public ASR loading can resolve a Granite Speech model from a local path whose `config.json` uses `model_type: granite_speech`.
- Runtime inference accepts a mono 16 kHz waveform or an audio path through the ASR adapter and returns `ASROutput(text=..., language=...)`.
- The model path is local-first and uses `.safetensors` state dicts. Original upstream shards must load through explicit config parsing and deterministic key handling.
- The inference pipeline follows the verified reference shape:
  `audio -> 80-bin HTK log-mel -> pair-stack to 160 dim -> 16-layer Conformer CTC encoder -> 2-layer QFormer projector -> <|audio|> embedding replacement -> Granite causal LM -> greedy decode`.
- The Granite LM, KV cache, causal mask, logits scaling, tokenizer wrapper, prompt rendering, feature extraction, and generation loop are implemented inside this repository or reused from existing `mlx-speech` code.
- The loader accounts for original checkpoint sanitizer requirements: drop `num_batches_tracked`; transpose original 1D convolution weights for `up_conv`, `down_conv`, and `depth_conv` when needed.
- The first supported generation mode is deterministic greedy transcription. Sampling, streaming, speculative decoding, and translation prompts are optional follow-ups unless needed for parity smoke.
- The core generation pipeline stays side-effect-free and fits the existing ASR surfaces. Local transcript artifacts for smoke and quality inspection are written only by tests or scripts into `outputs/granite_speech_asr/`.
- A local diagnostic transcription script can run Granite ASR over selected existing WAV files under `outputs/` and write one transcript per input plus a small summary without moving, rewriting, or mixing with the existing source audio and smoke artifacts.

## Constraints

- Pure MLX runtime only. No `torch`, `torchaudio`, `mlx_lm`, `transformers`, `vllm`, or `mlx-audio` imports in `src/mlx_speech`.
- `.references/` remains read-only reference material; do not vendor reference code wholesale.
- Upstream Hugging Face weights stay under `models/`, never under `src/` or git.
- Preserve the existing ASR architecture boundary: `asr/` dispatches/adapts; family internals live under `models/<family>/`; orchestration lives under `generation/<family>.py`.
- Do not broaden into Granite Speech 4.1, NAR, training, LoRA fitting, ONNX, server APIs, diarization, timestamps, or benchmarking unless a later spec says so.
- Runtime dependencies should stay within the current project stance. Any new dependency requires evidence that local implementation is not practical.

## Risks

- The `mlx-audio` reference relies on `mlx_lm` for Granite LM layers, cache, mask, sampling utilities, and generation. A local implementation must replace that boundary, not wrap it.
- The reference relies on `transformers.AutoTokenizer`; this repo must render the small chat template and use local tokenizer assets instead.
- Feature extraction details affect WER and may drift if mel scaling, STFT padding, log compression, or pair stacking differ from the reference.
- The original checkpoint is BF16 and about 4.3 GB. A working first pass may be memory-heavy until an explicit MLX quantized conversion exists.
- The checkpoint key namespace mostly matches the reference implementation, but strict load parity still depends on exact module naming, Conv1d weight orientation, and skipped BatchNorm bookkeeping tensors.

## Acceptance Criteria

- `mlx_speech.asr.load(<local granite original path>)` returns a Granite ASR adapter when `config.json` has `model_type: granite_speech`.
- A Granite runtime can load the downloaded local original checkpoint with strict accounting: all required weights consumed, only intentional skipped keys allowed, and no silent missing model parameters.
- Unit tests cover config parsing, registry dispatch, tokenizer/audio-token prompt construction, audio-token embedding replacement, feature extraction shape, checkpoint key sanitizer behavior, and greedy decode loop behavior on tiny/fake modules.
- Checkpoint tests inspect the real local Granite original shards when present and skip cleanly when absent.
- Runtime smoke test, gated behind local checkpoint availability, transcribes `models/ibm/granite_4_0_1b_speech/original/multilingual_sample.wav` or a small local fixture into non-empty text without torch/transformers/mlx_lm imports.
- A local diagnostic command can transcribe existing generated samples from `outputs/` and write results under `outputs/granite_speech_asr/`, with non-empty transcript checks and per-file status reporting. This is a functional quality inspection, not a WER benchmark or exact-text gate.
- `pytest tests/unit/` passes before reporting implementation complete.
- Public docs or README model list state Granite Speech support only after runtime smoke passes.

## Anti-Goals

- Do not implement training, fine-tuning, LoRA fitting, dataset code, or model evaluation dashboards.
- Do not add torch-backed fallback paths under an MLX label.
- Do not route runtime inference through `mlx-audio`, `mlx_lm`, `transformers`, vLLM, or ONNX.
- Do not download weights at runtime as part of model construction.
- Do not make this a generic speech-language framework abstraction; keep it a family adapter consistent with the existing ASR surface.

## Deferred / Not In Scope

- Granite Speech 4.1 2B and NAR variants are deferred. This spec is for `granite-4.0-1b-speech`, because it is the model cloned and downloaded locally.
- MLX int8 or 4-bit conversion is deferred unless needed to make baseline inference possible. The first target is original BF16 checkpoint loading.
- Streaming, sampling controls, speculative decoding, and speech translation prompts are deferred after deterministic ASR works.
- End-to-end WER benchmarking against OpenASR is deferred; this spec requires local functional correctness and focused parity checks, not leaderboard reproduction.

## Assumptions

- The downloaded local checkpoint remains available at `models/ibm/granite_4_0_1b_speech/original/`.
- The model can fit for a local smoke on the target Apple Silicon machine in BF16. If not, the plan must add a conversion/quantization slice before runtime smoke.
- The existing `tokenizers` dependency can represent the Granite GPT-2 BPE tokenizer assets well enough for prompt encoding and decode.
