# Audio Examples

Sample outputs generated locally on Apple Silicon using mlx-speech.

## vibevoice_4speaker_convo.wav

Four-speaker conversation generated with VibeVoice. Demonstrates natural
multi-speaker turn-taking and voice variation at longer form lengths.

VibeVoice uses `Speaker N:` tags to distinguish voices. A 4-speaker prompt looks like:

```bash
python scripts/generate_vibevoice.py \
  --text "Speaker 1: Have you tried the new mlx-speech library? Speaker 2: Not yet, what does it do? Speaker 3: It runs TTS models locally on Apple Silicon. Speaker 4: No cloud, no PyTorch — sounds good to me." \
  --output outputs/vibevoice_4speaker_convo.wav
```
