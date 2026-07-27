# v7 — Nemotron 3.5 ASR (streaming, 0.6B)

Port NVIDIA's cache-aware streaming FastConformer-RNNT to a pure MLX runtime.
This is the repo's first **streaming** ASR path and its first **transducer**
decoder. Both are reusable beyond this one checkpoint.

- Upstream weights: `nvidia/nemotron-3.5-asr-streaming-0.6b` (OpenMDW-1.1)
- NeMo class: `EncDecRNNTBPEModelWithPrompt`
- References: `.references/NeMo` (source truth), `.references/mlx-audio` (MLX prior art)

## Why This Model

Every ASR path in this repo today is offline: the encoder sees the whole
utterance, so the latency floor is the length of the audio. Nemotron 3.5 is
trained to be causal, so it emits text while audio is still arriving. At 600M
parameters it is small enough to run comfortably on Apple Silicon, and one
checkpoint covers 40 language-locales.

Scope is multilingual only. The English-only sibling is out of scope.

---

## Architecture

```
waveform 16 kHz
   │
   ├─ 1. Mel front end ─────────► 128 mel bins @ 100 Hz   (10 ms / frame)
   │
   ├─ 2. Causal 8x subsampling ─► d_model 1024 @ 12.5 Hz  (80 ms / frame)
   │
   ├─ 3. FastConformer encoder ─► 24 layers, bounded lookahead
   │
   ├─ 4. Language prompt fusion ► one-hot 128 ⊕ 1024 → MLP → 1024
   │
   └─ 5. RNN-T greedy decode ───► text
```

### 1. Mel front end

NeMo `AudioToMelSpectrogramPreprocessor`. Parameters that must match exactly, or
every downstream number drifts:

| | value | note |
| --- | --- | --- |
| `sample_rate` | 16000 | |
| `features` | 128 | mel bins |
| `n_fft` | 512 | |
| `window_size` | 0.025 s | 400 samples |
| `window_stride` | 0.01 s | 160 samples → **10 ms per frame** |
| `preemph` | 0.97 | first-order high-pass before framing |
| `dither` | 1e-5 | tiny noise, prevents log(0) |
| `log_zero_guard_value` | 2^-24 | added inside the log |
| `normalize` | `NA` | **no** per-feature normalization |

The `normalize: NA` is the trap. Most ASR front ends normalize per utterance.
This one does not, so the encoder's first layer expects raw log-mel scale.

### 2. Causal depthwise-striding subsampling

Three stride-2 conv stages reduce the frame rate 8x, from 100 Hz to 12.5 Hz.
The first is a full conv, the next two are depthwise-plus-pointwise, which is
what makes it "Fast"Conformer rather than a plain Conformer.

**Why subsample at all.** Attention is O(T²). At 100 Hz a 30-second clip is 3000
frames. At 12.5 Hz it is 375. That is a 64x reduction in attention cost, and
speech simply does not carry 100 independent decisions per second.

**Why causal.** Padding is asymmetric: `left = kernel - 1 = 2`, `right = stride - 1 = 1`.
A standard conv pads symmetrically and therefore mixes in future frames. This one
does not, which is the first of three reasons the model can stream.

The 8x factor is the single most important number in the model:

```
10 ms per mel frame × 8 = 80 ms per encoder frame
```

Every latency figure in NVIDIA's table is a multiple of that one number.

### 3. FastConformer encoder — 24 layers, d_model 1024, 8 heads

A Conformer block is a sandwich. Two half-step feed-forwards wrap a self-attention
and a convolution:

```
x = x + 0.5 · FFN₁(norm(x))          # half-step FFN, SiLU, expansion 4
x = x +       Attn(norm(x))          # rel-pos MHA, global context
x = x +       Conv(norm(x))          # causal depthwise k=9, local context
x = x + 0.5 · FFN₂(norm(x))          # half-step FFN
x = norm(x)
```

The design intent is division of labour. **Attention** carries long-range
context, which is what disambiguates homophones and carries speaker and topic
information. **Convolution** carries local context, the formant transitions and
phoneme boundaries that live in a 50 ms window. Transformers alone are weak at
the local structure; convnets alone are weak at the long range. Conformer takes
both.

Notable checkpoint quirks, all of which the port must honour:

- `use_bias=False` on every projection in the block.
- Attention is Transformer-XL style relative position, with **untied** per-layer
  `pos_bias_u` / `pos_bias_v`.
- The conv module's normalization is a **LayerNorm** but NeMo names the attribute
  `batch_norm`. Keep the name so checkpoint keys line up.
- `xscaling=False`, so no √d_model input scaling.

### 4. Language-ID prompt conditioning

A 128-dim one-hot language vector is broadcast across time, concatenated to the
1024-dim encoder output on the feature axis, and projected back down:

```
concat(enc[1024], one_hot[128]) → Linear(1152 → 2048) → ReLU → Linear(2048 → 1024)
```

This is why one 600M checkpoint covers 40 locales without model swapping. Passing
`auto` selects a prompt index that lets the model detect language itself and emit a
leading tag. Auto costs roughly 0.8 WER on the tier-1 average, concentrated in
non-Latin scripts, so language-specified is the better default.

### 5. RNN-T decoder — the part that makes streaming possible

This is the most important section, because it is where this model differs
fundamentally from every other ASR path in the repo.

**How Whisper-style models decode.** An attention encoder-decoder cross-attends
over the *entire* audio sequence for every output token. It can look anywhere in
the recording at any time. That expressiveness is exactly why it cannot stream:
it needs all the audio before emitting the first token.

**How a transducer decodes.** RNN-T walks time strictly forward and never
revisits a frame. It has three parts:

| part | input | role |
| --- | --- | --- |
| Encoder | audio | acoustic model — what sounds are present at frame *t* |
| Prediction network | tokens emitted so far | language model — what token is likely next. **Never sees audio.** |
| Joint network | both | combines them into a distribution over vocab + blank |

The joint is additive in a shared 640-dim projection:

```
joint(t, u) = Linear₆₄₀→₁₃₀₈₈( ReLU( Wₑ·enc[t] + Wₚ·pred[u] ) )
```

That defines a **T × U lattice**: time on one axis, emitted text on the other.
Training sums over all monotonic paths through it. Inference only ever needs one
cell at a time, which is what makes it cheap.

The vocabulary is 13,087 tokens **plus one blank**, and blank is the whole trick:

```python
while time < num_frames:
    joint_out = joint(enc[time], pred_state)
    token = argmax(joint_out)
    if token == BLANK:
        time += 1          # advance time, emit nothing
    else:
        emit(token)        # advance text, stay on the same frame
        pred_state = prediction_net(token)
```

Two pointers into the lattice. Blank moves right, a real token moves down. The
alignment is **monotonic**, so the decoder can never need a frame it has already
passed, and it never needs a frame it has not yet received. That property is what
"streaming-native" actually means.

Two practical consequences:

- `max_symbols = 10` caps tokens emitted at a single frame. Without it a degenerate
  model can emit forever without advancing time.
- Most frames are blank, so the LSTM and joint often run once per frame and move
  on. Skipping blank frames is where the large reported speedups come from.

Prediction network shape: embedding over `vocab + 1` (blank as pad) into a 2-layer
LSTM at 640 hidden. It is a small autoregressive LM whose only job is to make the
joint's next-token guess linguistically plausible.

---

## What Enables Live Capability

Three independent mechanisms. All three are required. Removing any one breaks
streaming.

### (a) Bounded lookahead — the `chunked_limited` attention mask

Encoder frames are grouped into non-overlapping chunks of `right_context + 1`
frames. A frame may attend to its own chunk and the previous
`left_context ÷ chunk_size` chunks. Nothing later. Verified against
`NeMo/nemo/collections/asr/modules/conformer_encoder.py:856-869`:

```python
chunk_size  = right_context + 1
left_chunks = left_context // chunk_size
chunk_idx   = arange(T) // chunk_size          # trunc division
diff        = chunk_idx[:, None] - chunk_idx[None, :]
visible     = (diff >= 0) & (diff <= left_chunks)
```

`att_context_size = [left, right]` is the accuracy-versus-latency dial, and every
published latency number falls straight out of the 80 ms frame period:

| `att_context_size` | chunk frames | latency | tier-1 WER |
| --- | --- | --- | --- |
| `[56, 0]` | 1 | 80 ms | 10.38 |
| `[56, 1]` | 2 | 160 ms | 10.00 |
| `[56, 3]` | 4 | 320 ms | 9.49 |
| `[56, 6]` | 7 | 560 ms | 9.12 |
| `[56, 13]` | 14 | 1.12 s | 8.84 |

Left context is 56 frames throughout, which is 4.48 seconds of history. The model
was trained on all five settings, so they are selectable at inference with no
retraining.

### (b) Causal convolution

Both the subsampling stack and the depthwise conv inside each Conformer block pad
left-only (`kernel - 1` left, `0` right). Convolution therefore contributes zero
lookahead. Without this, (a) would be pointless: the mask would block future
attention while the conv quietly leaked future frames anyway.

### (c) Cache-aware state — no recomputation

Each layer keeps two caches:

- **attention cache** — the last `left_context` attention *inputs*
- **conv cache** — the last `kernel - 1` post-GLU frames

A new chunk computes `Q` from the chunk alone and `K`/`V` from
`[cache ++ chunk]`. Because the cached window *is* precisely the allowed context,
**no attention mask is needed in the streaming path at all**. Subsampling is
incremental with a small mel cache.

This is the difference between cache-aware and buffered streaming. Buffered
streaming re-encodes an overlapping window every step and pays O(n²). Cache-aware
processes each frame exactly once and pays O(n). It is why a 0.6B model sustains
more concurrent streams than a 1.1B buffered one.

**The property that gives us a free correctness oracle:** because the cached
window equals the mask's allowed context, the streamed encoder output is
*frame-identical* to the offline `chunked_limited` encoder at native chunk size.
Not approximately. Bit-for-bit within numerical tolerance. Slice 5 tests exactly
this.

---

## Two Knobs, Not One

Blog coverage of this model routinely collapses two independent parameters into
one. Keep them separate in the API:

- **`att_context_size`** — lookahead. Real accuracy cost, ~1.5 WER across the
  range. This is a quality dial.
- **frame / chunk length** — how much audio is handed to the model per call. Zero
  accuracy cost, because cache-aware means each frame is processed once against
  identical context regardless of batching. This is purely a latency and
  throughput dial.

---

## Performance and Coherence Constraints

These apply to every slice, not just a final tuning pass.

### Coherence with the existing codebase

- Load through the shared `mlx_speech.asr.load(...)` entry point, same as
  `cohere-asr`, `qwen3-asr`, and `granite-speech-asr`. Local-path-first, alias
  second.
- Keep runtime and conversion separate. Model code in
  `src/mlx_speech/models/nemotron_asr/`, conversion in `scripts/`.
- Follow the module split the other ASR families already use: `config.py`,
  `encoder.py`, `checkpoint.py`, `feature_extraction.py`, `tokenizer.py`,
  `model.py`. A reader who knows `granite_speech_asr/` should find their way
  around this without a map.
- `granite_speech_asr/encoder.py` already has `DepthWiseConv1d`,
  `ConformerConvModule`, and relative-position `ConformerAttention`. Read it
  before writing a second Conformer. Share where the shapes genuinely match,
  and do not force-fit where they do not. Nemotron is causal and untied; Granite
  is neither.
- Streaming introduces a new public surface. Design it once and deliberately,
  since Voxtral Realtime and future transducers will reuse it.

### Performance

- **Cache-aware means O(n).** Any implementation that re-encodes overlapping
  windows has silently reverted to buffered streaming and lost the model's main
  advantage. The parity test in slice 6 catches correctness, not this. Measure
  work per frame and assert it is constant.
- **Preallocate the encoder caches.** Attention cache is
  `(1, left_context, 1024)` per layer and conv cache is `(1, kernel-1, 1024)`
  per layer, both fixed size. Ring buffers, not concatenate-and-slice, which
  would allocate per chunk across 24 layers on every step. This repo already
  learned that lesson in `fish_s2_pro/cache.py`.
- **Skip blank frames in the decode loop.** Most frames emit blank. The LSTM and
  joint should not run more than necessary, and `mx.eval` placement should not
  force a sync per frame.
- **Batch the joint carefully.** `joint()` broadcasts to a `T × U` lattice in
  training. Inference needs exactly one cell. Do not materialize the lattice.
- **Quantization from the start.** Default to int8 like the rest of the repo.
  The CoreML port reports ~55% size reduction at WER parity with mixed
  precision, so treat that as the target to match, not beat.
- Benchmark with `mx.get_peak_memory()` and RTFx against the mlx-audio
  reference. Being slower than the reference is a defect, not a tradeoff.

## Slices

Each slice lands independently testable, per the repo's working rules.

1. **References and plan** — pin NeMo + mlx-audio, write this document.
2. **Mel front end + causal subsampling** — gate: mel parity against NeMo within tolerance.
3. **FastConformer encoder** — gate: encoder output parity on fixed input.
4. **RNN-T prediction, joint, greedy decode** — gate: token-identical transcript.
5. **Language prompt conditioning** — gate: specified and `auto` both decode.
6. **Cache-aware streaming** — gate: streamed output frame-identical to offline.
7. **Checkpoint conversion** — gate: every key mapped, none silently dropped.

## Open Questions

- **License.** OpenMDW-1.1. Sources disagree on whether the NVIDIA Open Model
  License also applies. Read the LICENSE file in the model repo before publishing
  converted weights.
- **Batching.** mlx-audio's mask drops NeMo's padding-mask term because it only
  runs batch=1. Decide deliberately whether to inherit that limit or support
  batched inference with correct padding.
- **Quantization.** The CoreML port reports ~55% size reduction at WER parity
  using INT8 with 6-bit palettized middle layers. Worth testing whether the repo's
  existing int8 and mxfp8 paths reach the same.
