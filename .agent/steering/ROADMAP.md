# Roadmap

## Now

**Nemotron 3.5 ASR (streaming, 0.6B).** The repo's first streaming ASR path and
first transducer decoder. Every ASR family shipped so far is offline, so the
latency floor is the length of the audio. Cache-aware FastConformer-RNNT changes
that, and both the transducer decode loop and the cache-aware Conformer are
reusable beyond this checkpoint.

Multilingual only. The English-only sibling is out of scope.

## Next candidates

Ordered by effort-to-value, not preference. Nothing here is committed.

- **Granite Speech 4.1 2B.** Its `config.json` matches this repo's existing
  `granite_speech_asr` defaults on every field: 16-layer Conformer, 2-layer
  BLIP-2 Q-Former, `granite-4.0-1b-base` decoder, `window_size` 15,
  `downsample_rate` 5. A weights-only swap plus tokenizer and chat-template work
  for the added translation prompts. Cheapest item on the list.
- **Cohere Transcribe Arabic (07-2026).** Same Conformer encoder-decoder family
  as the shipped `cohere-asr`. Leads the Arabic leaderboard at 25.87 WER, beating
  a 7B model at 2B. Likely a remap plus tokenizer swap.
- **MOSS-Transcribe-Diarize 0.9B.** Transcript, speaker labels, word timestamps,
  and acoustic events in one generation over recordings up to 90 minutes, across
  50+ languages. Diarization is a capability the repo has none of, and no MLX
  port appeared to exist when surveyed.
- **ARK-ASR-3B.** Top of the Open ASR Leaderboard, 19 languages, Whisper-style
  encoder with MLP adapter into a Qwen decoder. A shape this repo already
  implements twice. Cost is reading their custom `arkasr` remote code closely.

## Deferred or not now

Nothing currently deferred.

## Closed

- **RE-USE reference-denoise.** Shipped 2026-06-21 as
  `2026-06-14-reuse-voice-ref-mlx`. Pure-MLX SEMamba with a hand-written
  selective scan, wired into DramaBox as opt-in `denoise_ref=True`. Weights
  published at `appautomaton/re-use-semamba-mlx` under NSCLv1 (non-commercial).
