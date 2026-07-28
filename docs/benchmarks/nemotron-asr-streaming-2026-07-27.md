# Nemotron 3.5 ASR streaming benchmark — 2026-07-27

## Setup

- Apple M5 Max MacBook Pro, 128 GB unified memory
- MLX bf16 checkpoint
- encoder-only cache-aware streaming, batch one
- each native encoder chunk synchronized, matching a live session
- median of nine measured runs after warm-up
- pinned mlx-audio `d28d68c6` reference

Reproduce:

```bash
python scripts/eval/benchmark_nemotron_asr.py \
  --mel-lengths 256 512 1024 \
  --repeats 9 \
  --left-context 56 \
  --right-context 13
```

## Default 1120 ms mode: `(56, 13)`

| Audio | Encoder frames | Runtime | RTFx | ms/frame | Incremental peak |
| ---: | ---: | --- | ---: | ---: | ---: |
| 2.56 s | 33 | mlx-speech | 22.134× | 3.505 | 823.3 MB |
| 2.56 s | 33 | mlx-audio | 23.003× | 3.372 | 827.7 MB |
| 5.12 s | 65 | mlx-speech | 28.091× | 2.804 | 840.4 MB |
| 5.12 s | 65 | mlx-audio | 28.393× | 2.774 | 845.7 MB |
| 10.24 s | 129 | mlx-speech | 28.775× | 2.759 | 846.5 MB |
| 10.24 s | 129 | mlx-audio | 28.809× | 2.755 | 851.5 MB |

At steady state, mlx-speech is within 0.12% of the reference's RTFx and uses
about 5 MB less incremental peak memory. The once-per-block counter separately
asserts exact O(n) work: `encoder_frames × 24`, with no overlapping-window
recomputation.

## 320 ms mode: `(56, 3)`

At 10.24 seconds / 129 encoder frames, mlx-speech measured 9.308× RTFx and
819.6 MB incremental peak; mlx-audio measured 9.637× and 828.4 MB. The fixed
cache writes cost 3.5% throughput in this tighter mode while saving about 8.8 MB.
This remains an optimization target; it does not change output tokens or the O(n)
work bound.

## Interpretation

The 2.56-second row includes fixed session/cache startup and is not a steady-state
throughput estimate. From 5.12 to 10.24 seconds, mlx-speech stays near 2.8 ms per
encoder frame and peak memory remains bounded rather than growing with utterance
length. These are single-stream encoder numbers, not NVIDIA's H100 concurrency
figures and not an end-to-end WER evaluation.
