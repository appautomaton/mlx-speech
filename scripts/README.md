# Scripts

This directory is for small maintenance helpers such as checkpoint inspection
and conversion utilities.

Current v0 scripts:

- `download_openmoss_v0_assets.py` — download the upstream local checkpoints
  into `models/openmoss/.../original/`
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
