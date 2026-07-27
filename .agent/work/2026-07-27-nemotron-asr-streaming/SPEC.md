# SPEC: Nemotron 3.5 ASR streaming (pure-MLX)

## Bounded goal

Port NVIDIA Nemotron 3.5 ASR (`nvidia/nemotron-3.5-asr-streaming-0.6b`, a
cache-aware streaming FastConformer-RNNT with language-ID prompt conditioning)
to a pure-MLX runtime, exposing both offline transcription and true cache-aware
streaming across the model's 40 language-locales.

## Broader intent

Every ASR family in this repo today is offline: the encoder sees the whole
utterance, so the latency floor is the length of the audio. This change adds the
first **streaming** ASR path and the first **transducer** decoder. Both are
reusable infrastructure. A cache-aware Conformer and an RNN-T greedy loop unlock
the Parakeet and Canary families afterward, and low-latency on-device
transcription is the use case Apple Silicon is actually for.

## Work scale and shape

- **Scale:** Capability-sized. New model port (architecture + weight conversion +
  numerical parity + a new public streaming surface). 600M params, comparable in
  shape to the Granite and Qwen3-ASR ports but with two subsystems the repo has
  never built.
- **Shape:** Parity port against a NeMo reference, plus a new streaming API.

## Selected lenses

`engineering` (primary: transducer decode + cache-aware state), `runtime`
(latency and allocation behavior on Apple Silicon, the whole point of the
change), `product` (which latency and language defaults we expose, and how
honestly we describe tier-2/3 language quality).

## Target stakeholder

mlx-speech users on Apple Silicon who need transcription while audio is still
arriving: live captioning, voice agents, dictation. Secondarily the maintainer,
since streaming is the capability that differentiates an MLX-native speech
library from a wrapper.

## Constraints (that change implementation)

- **Pure MLX runtime, no torch.** Torch and NeMo are permitted only for offline
  reference capture and conversion, never on the inference path.
- **The encoder must not see the future.** Three independent mechanisms are all
  required: the `chunked_limited` attention mask, causal convolution in both the
  subsampling stack and the Conformer conv module, and per-layer cache-aware
  state. Any one of them missing silently breaks streaming while still producing
  plausible offline transcripts.
- **Reference behavior to match** (`.references/NeMo`, `EncDecRNNTBPEModelWithPrompt`):
  the `chunked_limited` mask at
  `nemo/collections/asr/modules/conformer_encoder.py:856-869`; 128-mel front end
  with `normalize: NA` (no per-feature normalization, a common and silent
  mismatch); 8x causal depthwise-striding subsampling with asymmetric padding
  (left `k-1`, right `stride-1`) on both time and frequency; `use_bias=False`
  throughout the Conformer blocks; untied per-layer `pos_bias_u`/`pos_bias_v`;
  the conv module's normalization is a LayerNorm that NeMo names `batch_norm`.
- **Two knobs stay independent in the API.** `att_context_size` is lookahead and
  costs accuracy (~1.5 WER across the range). Frame/chunk length is latency and
  throughput only, and costs nothing, because cache-aware means each frame is
  processed once against identical context. Collapsing them is the single most
  common error in third-party descriptions of this model.
- **Cache-aware means O(n).** Any implementation that re-encodes overlapping
  windows has reverted to buffered streaming and lost the model's main advantage.
  Correctness tests will not catch this; it needs its own measurement.
- **Weights:** `nvidia/nemotron-3.5-asr-streaming-0.6b`, license **OpenMDW-1.1**.
  The HF repo tags `openmdw-1.1` and the card states the model is ready for
  commercial use, but sources disagree on whether the NVIDIA Open Model License
  also applies. Read the LICENSE in the model repo and confirm it permits
  redistributing a derivative (quantized) build before publishing.
  `.safetensors`; weights never in git.
- **We publish our own quantized build.** The `hf` CLI is authenticated, so
  originals download directly. Produce an MLX **int8** build as the default
  runtime weight, matching every other ASR family in this repo, plus **bf16** as
  the unquantized reference. Published under the `appautomaton` org.
- **Naming follows the current convention**, not the older one. Recent cards
  (`qwen3-asr-1.7b-int8-mlx`, `qwen3-asr-1.7b-bf16-mlx`) use `-int8-mlx` /
  `-bf16-mlx`; older ones use `-8bit-mlx`. Use the newer form:
  `appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx` and
  `appautomaton/nemotron-3.5-asr-streaming-0.6b-bf16-mlx`, preserving upstream's
  model name so provenance is obvious.
- **Coherence with existing ASR families.** Load through
  `mlx_speech.asr.load(...)`, local-path-first. Module split mirrors
  `granite_speech_asr/`. Read that package's existing Conformer primitives
  (`DepthWiseConv1d`, `ConformerConvModule`, relative-position
  `ConformerAttention`) before writing a second one, and share only where shapes
  genuinely match. Nemotron is causal and untied; Granite is neither.

## Risks

- **RNN-T is a new decoder family here.** Every other model in the repo uses a
  single autoregressive loop. A transducer is a nested time-and-token walk over a
  T×U lattice with a blank symbol that advances time. Primary implementation
  risk.
- **Cache-aware encoder state is new.** The repo's KV cache work
  (`fish_s2_pro/cache.py`) is decoder-side. Encoder-side conv ring buffers and
  attention left-context caches are unbuilt. Naive concatenate-and-slice across
  24 layers every chunk would allocate heavily, which is exactly the defect class
  this repo just fixed elsewhere.
- **Checkpoint format is unconfirmed.** The HF repo may ship a `.nemo` tarball, a
  transformers-compatible layout, or both. Resolve in plan before writing the
  converter.
- **Mel front-end drift.** `normalize: NA` plus `preemph`, `dither`, and
  `log_zero_guard_value` must match exactly. A small mismatch degrades WER
  without ever failing loudly.
- **Numerical details.** `rel_shift` in relative-position attention and the
  trunc-division chunk indexing in the mask are both easy to get subtly wrong and
  produce output that looks reasonable.
- **License ambiguity** blocks weight publication, though not the port itself.

## Required outcome

- **Behavior:** `mlx_speech.asr.load("nemotron-asr-streaming")` transcribes an
  audio file offline, and a streaming entry point yields incremental transcripts
  as audio arrives, both in pure MLX.
- **Streaming is real:** per-layer attention and conv caches with incremental
  subsampling, O(n) in audio length, latency independent of utterance length.
- **Parity target:** the streamed encoder output is frame-identical to the
  offline `chunked_limited` encoder at native chunk size (`right_context + 1`).
  This equivalence is a property of the architecture, not an approximation, and
  it is the change's strongest correctness gate.

## Acceptance criteria

1. **Unit:** mel front end matches the NeMo featurizer on a fixed waveform within
   tolerance, including the no-normalization behavior.
2. **Unit:** causal subsampling output length and values match the NeMo
   `_calc_length` recurrence and reference activations.
3. **Unit:** the `chunked_limited` mask matches NeMo's construction exactly across
   all five trained `att_context_size` settings.
4. **Checkpoint:** converted weights load with every key mapped, none missing,
   none extra, none silently dropped.
5. **Runtime:** greedy RNN-T decode produces a token-identical transcript to the
   reference on a fixed clip.
6. **Runtime (hard gate):** streamed encoder output is frame-identical to the
   offline encoder at native chunk size. Must run green, not skip.
7. **Runtime:** language-specified and `auto` prompt modes both decode correctly.
8. **Performance:** per-frame work is constant as audio length grows (O(n) check),
   and peak memory plus RTFx are recorded against the mlx-audio reference.
   Slower than the reference is a defect.
9. **Pure MLX:** no torch, NeMo, or transformers import on the inference path.
10. `pytest tests/unit/` is green.
11. **Quantization:** an MLX int8 build loads and transcribes, with WER on a
    fixed evaluation set within an agreed tolerance of the bf16 build. Size
    reduction and RTFx recorded. Quantization that costs accuracy beyond the
    tolerance is not shipped as the default.
12. **Publication:** `appautomaton/nemotron-3.5-asr-streaming-0.6b-int8-mlx` and
    `-bf16-mlx` are live, each carrying the upstream license, NVIDIA attribution,
    a model card matching house format, and honest per-tier language quality.
    `mlx_speech.asr.load(...)` aliases resolve to them.

## Anti-goals

- **No English-only variant.** `nemotron-speech-streaming-en-0.6b` is out of
  scope. Multilingual only.
- No fine-tuning or adaptation for the 8 tier-3 locales.
- No training code, loss, or the RNN-T forward-backward. Inference only.
- No beam search in this change. Greedy decode first; beam is a follow-up.
- Not a general streaming framework. The streaming surface is designed to be
  reusable, but only this model is implemented here.
- No quantization below int8 in this change. 4-bit and mixed-precision
  palettization are a follow-up, informed by what the int8 build measures.

## Scope coverage

- **Included:** mel front end, causal subsampling, cache-aware FastConformer
  encoder, RNN-T prediction/joint/greedy decode, language-ID prompt conditioning,
  cache-aware streaming path, checkpoint conversion, int8 + bf16 quantized
  builds, model cards, HF publication under `appautomaton`, docs.
- **Deferred / not in scope:** beam search with internal-LM subtraction,
  batched inference (mlx-audio's reference drops NeMo's padding mask because it
  only runs batch=1 — decide deliberately in plan), sub-int8 quantization.
- **Decided this conversation:** multilingual checkpoint only; Automaton is the
  planning system; performance and coherence with existing families are standing
  constraints on every slice, not a final pass; we publish our own int8 build
  rather than depending on a third-party conversion.

## Assumptions

- The five trained `att_context_size` values (`[56,0]`, `[56,1]`, `[56,3]`,
  `[56,6]`, `[56,13]`) are all selectable at inference with no retraining, per the
  model card. Default to `[56,13]` for offline accuracy.
- Language-specified prompting is the documented default, since `auto` costs
  roughly 0.8 WER on the tier-1 average and considerably more on non-Latin
  scripts.
- mlx-audio's implementation is correct where it agrees with NeMo. It has been
  spot-checked on the mask and found faithful, but it is prior art to learn from,
  not a specification. NeMo is source truth.
