# Granite Speech Long-Audio Benchmark - 2026-05-29

This benchmark captures the first post-fix long-audio run after replacing the
Granite Speech LM's explicit attention matrix path with MLX efficient
scaled-dot-product attention.

## Source

- Audio: LibriVox / Internet Archive, *The Three Bears of Porcupine Ridge*,
  chapter 6, "Tracked by a Catamount"
- Text: Project Gutenberg ebook 49465, matching chapter text
- Duration: `1069.17s` (`17:49`)
- License status: public domain source family
- Local artifacts: generated under `/tmp`; audio, chunk WAVs, transcripts, and
  raw summaries are intentionally not committed

## Command

```bash
tmpdir=$(mktemp -d /tmp/granite-long-audio-fixed.XXXXXX)
python scripts/eval/granite_speech_long_audio.py \
  --output-dir "$tmpdir" \
  --source three-bears-catamount \
  --chunk-seconds 120 \
  --max-new-tokens 350
```

## Result

| Metric | Value |
| --- | ---: |
| Audio duration | `1069.17s` |
| Chunks | `9` |
| Prompt tokens | `10854` |
| Generated tokens | `2807` |
| Total wall time | `40.51s` |
| Transcription wall time | `34.16s` |
| RTF | `0.0320` |
| RTFx | `31.30x` |
| Non-empty chunks | `9/9` |
| Peak MLX memory | `14.83 GiB` |
| Final active memory | `4,626,942,668 bytes` |
| Final cache memory | `0 bytes` |
| Reference words | `2744` |
| Hypothesis words | `2510` |
| Word error rate | `0.1418` |
| Word accuracy | `0.8582` |
| Reference coverage | `0.8827` |
| Hypothesis precision | `0.9649` |

## Interpretation

- The prior 100+ GB memory-pressure pattern was not reproduced in this run.
- Peak MLX memory stayed around `14.83 GiB`; the dominant steady-state active
  memory is model residency, not per-layer explicit `[heads, tokens, tokens]`
  attention matrices.
- The full `17:49` file still cannot be processed as one prompt because it
  exceeds the model context. Long audio requires context-safe chunking.
- Accuracy is useful but not final production quality: the chunked transcript
  had strong precision (`0.9649`) and good coverage (`0.8827`), with remaining
  gaps likely from chunk boundaries, greedy decoding, and generated-token caps.

