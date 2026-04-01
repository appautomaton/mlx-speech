# Scripts

This directory is for small maintenance helpers such as checkpoint inspection
and conversion utilities.

Current v0 scripts:

- `download_openmoss_v0_assets.py` — download the upstream local checkpoints
  into `models/openmoss/.../original/`
- `convert_moss_sound_effect.py` — convert the upstream `MOSS-SoundEffect`
  checkpoint into `models/openmoss/moss_sound_effect/mlx-4bit/`
- `inspect_moss_local_checkpoint.py` — inspect the original upstream speech
  checkpoint layout
- `validate_moss_local_alignment.py` — validate original checkpoint keys and
  shapes against the current MLX model tree
- `convert_moss_local.py` — convert the upstream `MossTTSLocal` checkpoint into
  `models/openmoss/moss_tts_local/mlx-int8/`
- `convert_moss_audio_tokenizer.py` — convert the upstream Cat codec checkpoint
  into `models/openmoss/moss_audio_tokenizer/mlx-int8/`
- `generate_moss_local.py` — run end-to-end text-to-waveform synthesis from
  local `mlx-int8` weights and save a WAV file
- `compare_moss_local_upstream_parity.py` — compare MLX runtime behavior
  against the upstream torch reference on processor packing, greedy rollout,
  and codec decode
- `convert_moss_ttsd.py` — convert the upstream `MossTTSDelay` / TTSD
  checkpoint into `models/openmoss/moss_ttsd/mlx-int8/`
- `generate_moss_sound_effect.py` — run MOSS-SoundEffect generation from
  local sound-effect weights; expects `models/openmoss/moss_sound_effect/mlx-4bit/`
  by default and supports explicit `--model-dir` / `--codec-dir` overrides
- `generate_moss_ttsd.py` — run TTSD generation / continuation from local
  TTSD weights; uses `mlx-int8` by default and supports explicit `--model-dir`
  / `--codec-dir` overrides for custom checkpoint paths
- `audit_step_audio_editx_checkpoint.py` — inspect the local Step-Audio-EditX
  Step1 checkpoint config, tensor prefixes, and checkpoint/model alignment
- `audit_step_audio_tokenizer_assets.py` — inspect the local Step-Audio text
  tokenizer assets plus the external dual-tokenizer asset directory and sample
  prompt packing
- `generate_step_audio_editx.py` — run local Step-Audio clone/edit waveform
  generation from prompt audio, prompt text, and target/edit input
- `convert_step_audio_tokenizer.py` — package the local Step-Audio dual-tokenizer
  runtime assets into another directory and validate both tokenizer paths there
- `convert_step_audio_editx.py` — convert the original Step-Audio Step1
  checkpoint into a local `mlx-int8` runtime directory and copy the supporting
  runtime assets
- `run_local_speech_smoke_eval.py` — run a small sequential local smoke-eval
  across Cohere ASR, Moss local, TTSD, and VibeVoice, then write generated
  artifacts plus back-transcribed summaries under `outputs/tests/`
- `materialize_clone_eval_macos.py` — generate the fixed English clone-eval
  reference set from macOS built-in voices
- `sweep_clone_presets.py` — run clone preset sweeps over the fixed eval set
  and save WAV outputs plus a JSON summary
