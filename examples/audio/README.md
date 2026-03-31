# Audio Examples

Sample outputs generated locally on Apple Silicon using mlx-speech.

## vibevoice_4speaker_convo.wav

Four-speaker conversation generated with VibeVoice. Demonstrates natural
multi-speaker turn-taking and voice variation at longer form lengths.

VibeVoice uses `Speaker N:` tags to distinguish voices — up to 4 speakers in a single pass.

```bash
TEXT="Speaker 1: {line for speaker 1} \
Speaker 2: {line for speaker 2} \
Speaker 3: {line for speaker 3} \
Speaker 4: {line for speaker 4}"

python scripts/generate_vibevoice.py \
  --text "$TEXT" \
  --output outputs/vibevoice_4speaker_convo.wav
```
