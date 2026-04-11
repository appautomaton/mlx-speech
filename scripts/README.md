# scripts/

Repo-local entry points for conversion, generation, evaluation, and audit.
Run scripts directly from the repo root — `python scripts/<subdir>/<name>.py`.

## Layout

| Subdir | Purpose |
|--------|---------|
| `convert/` | Convert upstream checkpoints to MLX format. Run once per model before generating. |
| `generate/` | TTS synthesis, ASR transcription, and batch generation. |
| `eval/` | Smoke eval suite, clone benchmarks, and quality sweeps. |
| `audit/` | Checkpoint inspection, parity checks, and alignment validation. |
| `hugging_face/` | HuggingFace Hub upload helpers — release workflow, not runtime. |

## convert/

| Script | Model |
|--------|-------|
| `fish_s2_pro.py` | Fish S2 Pro |
| `longcat_audiodit.py` | LongCat AudioDiT |
| `vibevoice.py` | VibeVoice Large |
| `moss_local.py` | OpenMOSS TTS Local |
| `moss_audio_tokenizer.py` | OpenMOSS Audio Tokenizer (shared codec) |
| `moss_ttsd.py` | MOSS-TTSD |
| `moss_sound_effect.py` | MOSS-SoundEffect |
| `step_audio_tokenizer.py` | Step-Audio tokenizer |
| `step_audio_editx.py` | Step-Audio-EditX |
| `cohere_asr.py` | CohereASR |
| `download_openmoss_v0_assets.py` | Download OpenMOSS v0 upstream assets (prerequisite to moss_* conversions) |

## generate/

| Script | Model | Notes |
|--------|-------|-------|
| `fish_s2_pro.py` | Fish S2 Pro | |
| `longcat_audiodit.py` | LongCat AudioDiT | |
| `batch_longcat_audiodit.py` | LongCat AudioDiT | Batch/JSONL mode |
| `vibevoice.py` | VibeVoice Large | |
| `moss_local.py` | OpenMOSS TTS Local | |
| `moss_ttsd.py` | MOSS-TTSD | |
| `moss_sound_effect.py` | MOSS-SoundEffect | |
| `step_audio_editx.py` | Step-Audio-EditX | |
| `cohere_asr.py` | CohereASR | Transcription (ASR) |

## eval/

| Script | Purpose |
|--------|---------|
| `run_local_speech_smoke_eval.py` | End-to-end TTS+ASR smoke suite |
| `materialize_clone_eval_macos.py` | Materialize voice-clone eval assets on macOS |
| `sweep_clone_presets.py` | Sweep sampling presets for clone quality |
| `benchmark_moss_ttsd.py` | MOSS-TTSD latency and throughput benchmark |

## audit/

| Script | Purpose |
|--------|---------|
| `moss_ttsd_upstream_parity.py` | MOSS-TTSD vs upstream parity check |
| `step_audio_editx_checkpoint.py` | Step-Audio-EditX checkpoint audit |
| `step_audio_tokenizer_assets.py` | Step-Audio tokenizer asset audit |
| `inspect_moss_local_checkpoint.py` | Inspect OpenMOSS TTS Local checkpoint |
| `validate_moss_local_alignment.py` | Validate OpenMOSS weight alignment |
| `compare_moss_local_upstream_parity.py` | Compare OpenMOSS vs upstream parity |
