# Scripts

This directory holds repo-local conversion, audit, generation, transcription,
and evaluation entry points.

Current script groups:

- `OpenMOSS`: `download_openmoss_v0_assets.py`,
  `inspect_moss_local_checkpoint.py`, `validate_moss_local_alignment.py`,
  `compare_moss_local_upstream_parity.py`,
  `audit_moss_ttsd_upstream_parity.py`, `benchmark_moss_ttsd.py`,
  `convert_moss_local.py`, `convert_moss_audio_tokenizer.py`,
  `convert_moss_ttsd.py`, `convert_moss_sound_effect.py`,
  `generate_moss_local.py`, `generate_moss_ttsd.py`,
  `generate_moss_sound_effect.py`
- `VibeVoice`: `convert_vibevoice.py`, `generate_vibevoice.py`
- `CohereASR`: `convert_cohere_asr.py`, `transcribe_cohere_asr.py`
- `Step-Audio`: `audit_step_audio_editx_checkpoint.py`,
  `audit_step_audio_tokenizer_assets.py`, `convert_step_audio_tokenizer.py`,
  `convert_step_audio_editx.py`, `generate_step_audio_editx.py`
- `Evaluation/helpers`: `run_local_speech_smoke_eval.py`,
  `materialize_clone_eval_macos.py`, `sweep_clone_presets.py`

`scripts/hugging_face/` contains release upload helpers rather than runtime
entry points.
