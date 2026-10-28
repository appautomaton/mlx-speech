# Audio8-ASR-Infinite

Audio8-ASR-Infinite is a streaming Chinese/English speech recognizer from
Edge0 (Apache-2.0, about 4B parameters). This port runs the complete inference
path in MLX: waveform to log-mel features, a causal Voxtral-Realtime audio
tower, a projector, and a delay-conditioned Qwen2 decoder that emits one token
per audio clock.

The converted packages are:

```text
models/edge0/audio8_asr_infinite/
  original/   # upstream Hugging Face snapshot
  mlx-bf16/   # 8.2 GB
  mlx-int8/   # 4.4 GB, affine, group size 64
```

## Load and transcribe

```python
import mlx_speech

model = mlx_speech.asr.load("models/edge0/audio8_asr_infinite/mlx-bf16")
result = model.generate("speech.wav", language="zh", target_delay_ms=480)
print(result.language, result.text)
```

`language` is required and must be exactly `"zh"` or `"en"`. Audio arrays must
be mono 16 kHz float samples; file inputs are loaded, mixed down, and resampled
by the adapter. Decoding is greedy with EOS suppressed.

## Live session

```python
session = model.stream_session(language="en", frame_len=4, target_delay_ms=480)

for pcm in microphone:
    for token_id in session.feed(pcm):
        consume(token_id)

for token_id in session.finalize():
    consume(token_id)

result = session.result()
```

`feed()` accepts any number of 16 kHz samples and returns the visible token ids
decoded so far. `finalize()` is required: it appends the right padding
(`delay + 1 + right_pad_text_tokens` clocks of silence) that flushes the text
still trailing the audio, and returns those final token ids. Arbitrarily ragged
feeds are tested to produce the same tokens as a single feed.

`result()` returns `text`, `tokens`, `language`, `generated_token_ids`, and
`semantic_vad_probabilities`. It decodes the whole history, so call it once at
the end; a live display should accumulate the ids from `feed()` and
`finalize()`.

## Clock and delay

`frame_len` selects the audio clock: one text decision per `20 ms * frame_len`.
`target_delay_ms` is how far the text may trail the audio and must be a positive
multiple of the clock. The decoder is conditioned on both.

| `frame_len` | Clock | Left pad (tokens) | Post-trained `target_delay_ms` |
| ---: | ---: | ---: | --- |
| 4 | 80 ms | 18 | 240 / 320 / 480 / 560 |
| 6 | 120 ms | 12 | 240 / 480 |
| 8 | 160 ms | 9 | 320 / 480 |

Other multiples of the clock run, but the checkpoint was not post-trained for
them. The first prefill must fit the 750-frame attention window, which caps
`target_delay_ms` at about 13.5 s; larger values raise `ValueError`. Defaults are
`frame_len=4`, `target_delay_ms=480`.

## Unlimited-length audio

The checkpoint's native context is 30 seconds (1500 audio frames). A rolling
window keeps both KV caches, and with them the per-clock cost, constant beyond
that:

- The text KV cache is trimmed to `rolling_context_tokens` (default 375, 30 s at
  the 80 ms clock) every `rolling_trim_interval_ms` (default 3000). The first 16
  tokens are kept, and the surviving keys are re-rotated so RoPE positions stay
  contiguous.
- The audio KV cache is rebased to the sliding window (750 frames) whenever its
  position reaches the native limit.

With `rolling_context_tokens=0` the window is disabled and a session raises
`RuntimeError` once the native audio context is exhausted. Below the window
size, rolling and unbounded sessions produce identical tokens.

The session itself keeps every generated token id and one VAD probability array
per clock, which grows by roughly 20 MB per hour of audio.

## Semantic VAD

Four classifiers, one per horizon (0.5, 1, 2, and 3 s), each over 8 classes.
Upstream defines class 0 as end-of-turn. Every clock appends one `(4, 8)`
probability array to `semantic_vad_probabilities`; the text path does not
consume it.

## Conversion

```bash
python scripts/convert/audio8_asr_infinite.py --quant bf16
python scripts/convert/audio8_asr_infinite.py --quant int8
```

The converter reads the upstream safetensors without torch, merges
`semantic_vad_heads.safetensors`, transposes the two Conv1d weights into MLX
layout, and rejects any missing or unexpected tensor name or shape against the
config. The LM head is tied to the token embedding; upstream ships no separate
head. The int8 package quantizes the attention and MLP projections of both
stacks plus the token embedding, which doubles as the LM head; convolutions,
projector, norms, and VAD heads stay in bf16.

## Tests

```bash
pytest tests/unit/ -k audio8
pytest tests/runtime/test_audio8_runtime.py   # needs both local checkpoints
```
