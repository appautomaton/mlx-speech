# Audio Examples

Sample outputs generated locally on Apple Silicon using mlx-speech.

## vibevoice_4speaker_convo.wav

Four-speaker conversation generated with VibeVoice. Demonstrates natural
multi-speaker turn-taking and voice variation at longer form lengths.

VibeVoice uses newline-separated `Speaker N:` tags to distinguish voices — up to
4 speakers in a single pass.

```bash
TEXT=$'Speaker 1: {line for speaker 1}\n\
Speaker 2: {line for speaker 2}\n\
Speaker 3: {line for speaker 3}\n\
Speaker 4: {line for speaker 4}'

python scripts/generate/vibevoice.py \
  --text "$TEXT" \
  --output outputs/vibevoice_4speaker_convo.wav
```
