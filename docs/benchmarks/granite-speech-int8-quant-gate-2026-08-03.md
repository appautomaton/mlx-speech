# Granite Speech int8 quantization gate — 2026-08-03

**Verdict: PASS**

This gate compares the original IBM Granite Speech 4.0 1B BF16 checkpoint with
the `mlx-speech` selective-int8 artifact on an Apple M5 Max with 128 GB unified
memory. Each precision ran in a separate process with one excluded warmup and
five measured requests per sample. Both paths used greedy decoding,
`max_new_tokens=128`, and the same pre-release `mlx-speech` source tree.

The int8 conversion applies affine 8-bit weight quantization with group size 64
to 282 Granite causal-LM Linear and Embedding modules. No encoder or projector
module is quantized. The resulting weight file is byte-identical to the existing
MLX community conversion of the same upstream checkpoint.

## Artifact and memory

| Metric | BF16 | Selective int8 | Change |
| --- | ---: | ---: | ---: |
| Weight bytes | 4,626,527,776 | 2,904,308,838 | -37.225% |
| Loaded MLX active bytes | 4,626,778,828 | 2,904,558,284 | -37.223% |
| Process peak physical footprint | 8,581,698,336 | 6,763,122,264 | -21.191% |

The released `model.safetensors` SHA-256 is
`cf355a69e931ccac95d5cf942c3d540ba2456f06ad89c379c8132875b9098e6c`.

## Warm inference

The two 48 kHz mono inputs were generated from the same English project text
with Hank and Peggy voice references. RTF is wall time divided by audio time;
lower is better. Request peak is the median absolute MLX allocator peak across
five measured runs.

| Sample | Audio | BF16 median | int8 median | Time change | BF16 RTF | int8 RTF | Request peak change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hank | 12.16 s | 0.833 s | 0.734 s | -11.823% | 0.0685 | 0.0604 | -23.469% |
| Peggy | 12.80 s | 0.825 s | 0.689 s | -16.428% | 0.0644 | 0.0538 | -23.435% |

## Transcript comparison

All measured runs were deterministic within each precision. Int8 differed from
BF16 by one normalized word on each generated sample:

- Hank: BF16 emitted `mlxspeech`; int8 emitted `mxspeech`. Both scored 10.714%
  WER against the supplied project text.
- Peggy: BF16 emitted `mlx`; int8 emitted `mx`. WER changed from 3.571% to
  7.143%, one additional normalized word error.

On IBM's bundled multilingual sample, the first 64 generated tokens differed
only at the first word (`for` versus `but`); both retained the required
`timothy was a spoiled cat` phrase and the same following English/French text.
The artifact therefore passes the release gate with a disclosed minor
quantization difference; it is not transcript-identical to BF16.

## Test evidence

- `914 passed` in `tests/unit/`
- `70 passed, 34 skipped` in `tests/checkpoint/ tests/runtime/`
- strict int8 checkpoint alignment passed
- BF16 and int8 end-to-end bundled-sample runtime smokes passed
- an isolated install of the built 0.5.1 wheel loaded the quantized module tree
  and transcribed the bundled sample
- scoped Ruff and `git diff --check` passed
