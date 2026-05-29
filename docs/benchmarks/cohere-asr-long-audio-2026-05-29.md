# Cohere ASR Long-Audio Benchmark - 2026-05-29

This benchmark uses the same public-domain source as the Granite Speech
long-audio check: LibriVox audio for *The Three Bears of Porcupine Ridge*,
chapter 6, "Tracked by a Catamount", scored against the matching Project
Gutenberg chapter text.

## Recommended Quality Settings

For English prose/audiobook transcription:

- `language="en"`
- punctuation enabled
- ITN disabled (`itn=False`) unless numeric normalization is specifically needed
- greedy decoding, matching the current runtime surface
- `max_new_tokens=448` is enough for this source; the benchmark used `768` as a
  no-truncation guard, and no chunk came close to that cap

Cohere ASR accepts one long waveform at the API level, but internally chunks it:

- `max_audio_clip_s=35.0`
- `overlap_chunk_s=5.0`
- effective split threshold: about `30s`, with energy-based boundary search
- encoder position cap: `5000`
- decoder position cap: `1024`

## Command

The normal transcript-only path can be run as:

```bash
tmpdir=$(mktemp -d /tmp/cohere-long-audio.XXXXXX)
python scripts/generate/cohere_asr.py \
  --model-dir models/cohere/cohere_transcribe/mlx-int8 \
  --audio /path/to/tracked_by_a_catamount.mp3 \
  --language en \
  --max-new-tokens 448 \
  --output "$tmpdir/transcript.txt"
```

The metrics below used an equivalent `/tmp` one-off harness to also record
per-chunk timings, MLX memory snapshots, and reference-alignment anchors.

## Result

| Metric | Value |
| --- | ---: |
| Audio duration | `1069.17s` |
| Internal chunks | `34` |
| Generated tokens | `3800` |
| Token-cap chunks | `0` |
| Non-empty chunks | `33/34` |
| Total wall time | `18.94s` |
| Transcription wall time | `17.50s` |
| RTF | `0.0164` |
| RTFx | `61.11x` |
| Peak MLX memory | `5.25 GiB` |
| Final active memory | `4,132,240,827 bytes` |
| Final cache memory | `0 bytes` |
| Reference words | `2744` |
| Hypothesis words | `2795` |
| Word error rate | `0.0313` |
| Word accuracy | `0.9687` |
| Reference coverage | `0.9916` |
| Hypothesis precision | `0.9735` |

## Drift Check

The benchmark aligned each decoded chunk against the normalized reference text.
Chunks covering the actual story body advanced monotonically from reference word
`0` through `2744`.

Two chunks were unanchored:

- chunk `0`: LibriVox intro and title metadata, not present in the extracted
  Gutenberg chapter body
- chunk `33`: final short tail, decoded empty

There was no substantive mid-story drift, backward jump, or repeated-section
failure in this run.
