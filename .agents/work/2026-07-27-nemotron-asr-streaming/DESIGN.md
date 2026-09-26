# DESIGN: Nemotron 3.5 ASR streaming (pure-MLX)

Architecture and porting strategy. Contract: `SPEC.md`. Slices: `PLAN.md`.

## Signal path

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

## The number everything derives from

The subsampling factor is 8 and the mel hop is 10 ms, so one encoder frame is
**80 ms**. Every published latency figure is `(right_context + 1) × 80 ms`:

| `att_context_size` | chunk frames | latency | tier-1 WER |
| --- | --- | --- | --- |
| `[56, 0]` | 1 | 80 ms | 10.38 |
| `[56, 1]` | 2 | 160 ms | 10.00 |
| `[56, 3]` | 4 | 320 ms | 9.49 |
| `[56, 6]` | 7 | 560 ms | 9.12 |
| `[56, 13]` | 14 | 1.12 s | 8.84 |

NVIDIA's five "latency modes" are five right-context values. Left context is 56
frames throughout, which is 4.48 seconds of history.

## Components and file layout

Mirrors `granite_speech_asr/` so a reader familiar with that package can navigate
this one without a map.

```
src/mlx_speech/models/nemotron_asr/
  config.py             # dataclasses mirroring NeMo model_config.yaml
  feature_extraction.py # 128-mel featurizer, NeMo parity, normalize: NA
  subsampling.py        # causal dw-striding 8x, asymmetric pad
  attention.py          # rel-pos MHA (Transformer-XL), untied pos biases
  encoder.py            # FastConformer blocks + chunked_limited mask
  streaming.py          # per-layer attention/conv caches, incremental subsample
  transducer.py         # prediction net (LSTM), joint net, greedy decode
  prompt.py             # language-ID one-hot fusion
  checkpoint.py         # explicit weight remap, no silent fallbacks
  model.py              # assembly + public entry points

scripts/convert/nemotron_asr.py   # NeMo checkpoint -> safetensors
```

Conversion stays out of the runtime package, per the repo's separation rule.

## Encoder

24 Conformer blocks, `d_model` 1024, 8 heads, FFN expansion 4, `use_bias=False`.

Each block is a Macaron sandwich:

```
x = x + 0.5 · FFN₁(norm(x))     # half-step, SiLU
x = x +       Attn(norm(x))     # rel-pos, long-range context
x = x +       Conv(norm(x))     # causal depthwise k=9, local context
x = x + 0.5 · FFN₂(norm(x))     # half-step
x = norm(x)
```

Attention carries long-range context, which disambiguates homophones and tracks
speaker and topic. Convolution carries the local structure — formant transitions
and phoneme boundaries inside a 50 ms window — which attention models poorly.
Conformer takes both rather than choosing.

Checkpoint quirks the port must honour, each of which breaks weight loading or
numerics if missed:

- No bias on any block projection.
- Untied per-layer `pos_bias_u` / `pos_bias_v`.
- The conv module's norm is a **LayerNorm** that NeMo names `batch_norm`. Keep
  the attribute name so keys line up.
- `xscaling=False`, so no √d_model input scaling.

## RNN-T (the part that makes streaming possible)

An attention encoder-decoder cross-attends over the entire audio for every output
token, which is precisely why it cannot stream. A transducer walks time strictly
forward and never revisits a frame.

| Part | Input | Role | Shape |
| --- | --- | --- | --- |
| Encoder | audio | acoustic model | 1024 |
| Prediction network | tokens so far | LM, **never sees audio** | 2-layer LSTM, 640 |
| Joint network | both | vocab + blank | 640 → 13088 |

```
joint(t, u) = Linear₆₄₀→₁₃₀₈₈( ReLU( Wₑ · enc[t] + Wₚ · pred[u] ) )
```

That is a T×U lattice. Training sums over all monotonic paths; inference needs
one cell at a time. Do not materialize the lattice at inference.

The vocabulary is 13,087 tokens plus one blank, and blank is the mechanism:

```
while time < num_frames:
    token = argmax(joint(enc[time], pred_state))
    if token == BLANK:  time += 1                       # advance time
    else:               emit(token); pred_state = pred(token)   # advance text
```

Because the alignment is **monotonic**, the decoder can never need a frame it has
passed, and never needs a frame that has not arrived. That is what
"streaming-native" means architecturally.

`max_symbols = 10` caps emissions at one frame, guarding against a degenerate
loop that emits without advancing time.

## What enables live capability

Three mechanisms, all required.

**(a) Bounded lookahead.** Frames group into chunks of `right_context + 1`. A
frame attends to its own chunk and the previous `left ÷ chunk_size` chunks.
Verified against `NeMo/nemo/collections/asr/modules/conformer_encoder.py:856-869`:

```python
chunk_size  = right_context + 1
left_chunks = left_context // chunk_size
chunk_idx   = arange(T) // chunk_size          # trunc division
diff        = chunk_idx[:, None] - chunk_idx[None, :]
visible     = (diff >= 0) & (diff <= left_chunks)
```

**(b) Causal convolution.** Subsampling and the block conv both pad left-only.
Without this, (a) is theatre: the mask blocks future attention while convolution
leaks future frames anyway.

**(c) Cache-aware state.** Each layer holds the last `left_context` attention
inputs and the last `kernel - 1` post-GLU frames. A new chunk computes `Q` from
itself and `K`/`V` from `[cache ++ chunk]`.

Buffered streaming re-encodes an overlapping window each step and pays O(n²).
Cache-aware touches each frame once and pays O(n). That is why a 0.6B model
sustains 240 concurrent streams where a 1.1B buffered model manages 14.

## The free correctness oracle

Because the cached window **is** exactly the mask's allowed context, the
streaming path needs no attention mask at all. The window does the masking by
construction.

Consequence: streamed encoder output is frame-identical to offline encoder output
at native chunk size, within numerical tolerance. Not approximate. This is a far
stronger gate than "the transcript looks right", and it is the change's hard
gate (Slice 6, AC6).

## Cache allocation strategy

Both caches are **fixed size**, known at construction:

- attention cache: `(1, left_context, 1024)` per layer
- conv cache: `(1, kernel - 1, 1024)` per layer

Preallocate and write in place. Concatenate-and-slice across 24 layers on every
chunk allocates continuously, which is the defect class this repo fixed in
`fish_s2_pro/cache.py` (5.85 GB → 0.04 GB per token). Do not reintroduce it on
the encoder side.

## Parity strategy

Three tiers, cheapest first:

1. **Structural** (unit, no weights): mask construction, subsampling output
   lengths, and cache shapes against NeMo's formulas. Runs in CI.
2. **Numerical** (checkpoint/runtime, needs weights): encoder activations and
   decoded tokens against the mlx-audio reference on a fixed clip.
3. **Architectural invariant** (runtime, hard gate): streamed equals offline.
   This one needs no external reference at all, which makes it the most durable
   of the three.

## Open questions for the plan

- **Checkpoint format.** Confirm what `nvidia/nemotron-3.5-asr-streaming-0.6b`
  actually ships (`.nemo` tarball vs transformers layout) before writing the
  converter.
- **Batching.** mlx-audio drops NeMo's padding-mask term because it runs batch=1
  only. Inherit the limit deliberately, or implement padded batching.
- **Quantization.** The CoreML port reports ~55% size reduction at WER parity via
  INT8 with 6-bit palettized middle layers. Target matching that with the repo's
  existing int8 path.
