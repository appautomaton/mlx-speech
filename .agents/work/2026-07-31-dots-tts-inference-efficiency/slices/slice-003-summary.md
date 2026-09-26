# Slice 3 Execution Summary

## Status and decision

Complete. The all-BF16 recurrent attempt exceeded the established batch/stream waveform tolerance, so the accepted precision boundary keeps post-projection in BF16, SLSTM state and accumulation in FP32, and casts once into BF16 BigVGAN. Warm compiled vocoder execution is retained because its real four-patch median was 5.61% faster than eager.

## Changed files

- `src/mlx_speech/generation/dots_tts.py`
- `src/mlx_speech/models/dots_tts/audio_vae.py`
- `src/mlx_speech/models/dots_tts/vocoder.py`
- `tests/runtime/test_dots_tts_base.py`
- `tests/unit/test_dots_tts_audio_vae.py`
- `tests/unit/test_dots_tts_checkpoint_contract.py`
- `tests/unit/test_dots_tts_generation.py`
- `tests/unit/test_dots_tts_vocoder.py`
- `tests/unit/test_dots_tts_vocoder_streaming.py`

## Verification

- Focused Slice 3 unit command: 37 passed before the cache-lifetime fix; the final full unit suite includes the added regressions and passes 806 tests.
- Complete dots.tts base runtime file: 5 passed.
- Four-case fixed quality gate: passed. MF and SOAR WER regressions are `0.0`; speaker-cosine regressions are `0.002301` and `0.016507`, below the `0.02` threshold.
- Real MF 32-patch profile: batch `7.548089s`, RTF `1.474236`; stream `14.304717s`, TTFC `2.712254s`, RTF `2.793890`; both produced 245,760 finite, non-silent samples. Raw diagnostic SHA-256: `394a35c506ff2274e3d44e049b6e458c54eaa55997d3512a852e14ee51c1d176`.
- Real MF four-patch vocoder comparison after warmup: eager median `0.902430s`; compiled median `0.851787s`; compiled/eager `0.943882`. Spec review independently reproduced a comparable 5.4% improvement.
- Quality report SHA-256: `e673d808642393f4a5c0a1c123447fd692c32cfe3a1986a405179fa81912f753`.
- Scoped Ruff and `git diff --check`: passed.

## Reviews

- Spec compliance: `APPROVED`, no issues after the semantic-cache prerequisite and evidence were supplied.
- Code quality: `APPROVED`, no issues after adding live module-state compile inputs, a deterministic 12-entry LRU, explicit cache clearing, and correct real-test cleanup ordering.

## Unresolved risks or next action

Concurrent first-use compilation is not covered. Slice 4 proceeds with the approved serial request semantics and does not change the vocoder compile cache.
