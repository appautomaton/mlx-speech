# Upstream References

This repository uses `.references/` for optional local checkouts of upstream
projects that are useful for source inspection.

These checkouts are not part of the `mlx-speech` runtime, build, or packaging
story. They exist so implementation work can refer to upstream code locally
without turning those projects into vendored dependencies. Read-only; never
imported by the MLX runtime.

## Current Checkouts

- `.references/mlx`: Apple MLX. A real dependency of the published package; the
  checkout itself is for local source inspection only.
- `.references/mlx-audio`: MLX speech/audio community project. Reference for
  MLX-side implementation patterns, including Nemotron 3.5 ASR.
- `.references/MOSS-TTS`: Active OpenMOSS family repository. Best primary
  OpenMOSS reference point.
- `.references/MOSS-TTSD`: OpenMOSS delay-pattern dialogue TTS.
- `.references/dots.tts`: Official PyTorch source for dots.tts inference,
  fine-tuning, MeanFlow distillation, checkpoint behavior, and parity oracles.
- `.references/dots-tts-mlx`: Pure-MLX, inference-only dots.tts reference
  (SOAR and MeanFlow). Not a runtime dependency.
- `.references/FireRedTTS3`: Official PyTorch source for FireRedTTS3 Base and
  Instruct — RedAE, CAM++, conditioning, sampling, and waveform inference.
  Original weights live separately at `models/firered/firered_tts3/original/`
  (gitignored).
- `.references/fish-speech`: Fish Speech upstream. Source for the Fish S2 Pro
  port.
- `.references/Covo-Audio`: Covo Audio upstream, inspected during Fish S2 Pro
  bring-up.
- `.references/Step-Audio-EditX`: Step-Audio-EditX upstream. Voice cloning and
  audio editing reference.
- `.references/DramaBox`: Resemble's flow-matching diffusion TTS. Vendors a
  subset of LTX-2 (`ltx2/ltx_core`, `ltx2/ltx_pipelines`) as the diffusion
  framework.
- `.references/VibeVoice-ComfyUI`: ComfyUI integration for VibeVoice. Handy for
  cross-checking the VibeVoice pipeline wiring.
- `.references/meituan-longcat`: LongCat AudioDiT upstream. Diffusion TTS
  reference.
- `.references/Qwen3-ASR`: Qwen3-ASR source repo. Reference for the
  transformers/vLLM implementation, processor, prompt handling, streaming
  wrapper, and forced aligner. Code only; no weights.
- `.references/granite-4.0-1b-speech`: Hugging Face model repo for IBM Granite
  Speech 4.0 1B. Shallow clone with Git LFS smudge disabled; large model files
  remain as LFS pointers.
- `.references/transformers`: Sparse checkout of Hugging Face `transformers`,
  limited to `models/cohere_asr`, `models/moonshine`, and `models/parakeet`.
- `.references/NeMo`: NVIDIA NeMo, sparse clone limited to the ASR inference
  pipeline (`nemo/collections/asr/...`). Training code, configs, and weights
  are excluded. Source-truth for the Nemotron 3.5 ASR port. Key files:
  `modules/conformer_encoder.py` (cache-aware `chunked_limited` masking),
  `modules/rnnt.py`, `parts/submodules/rnnt_greedy_decoding.py`,
  `parts/preprocessing/features.py`.
- `.references/RE-USE`: NVIDIA RE-USE / SEMamba universal speech enhancement.
  Source-truth for the MLX RE-USE port (DramaBox `denoise_ref`). Code subset
  only; weights live gitignored at `models/reuse/original/`. License NSCLv1
  (non-commercial).
- `.references/mamba_ssm`: Two files only:
  `ops/selective_scan_interface.py` (`selective_scan_ref`, `mamba_inner_ref`)
  and `modules/mamba_simple.py` (`class Mamba`). The exact reference math the
  MLX selective-scan port mirrors, since `mamba_ssm` has no macOS wheels.

## Staged Weights and Assets

Not in `.references/`, but referenced alongside them (all gitignored):

- `models/nvidia/nemotron_3_5_asr_streaming_0_6b/original/`: The upstream repo
  ships both a 2.4 GB `.nemo` archive and a 2.6 GB Transformers
  `model.safetensors`; conversion uses the `.nemo` source. Governing terms are
  OpenMDW-1.1, not the NVIDIA Open Model License. Redistribution must retain
  the OpenMDW-1.1 text and applicable copyright/origin notices; an official
  license copy is staged beside the checkpoint.
- `models/stepfun/step_audio_editx/original/` and
  `models/stepfun/step_audio_tokenizer/original/`: Step-Audio assets for
  runtime bring-up, conversion, and source inspection.
- `models/firered/firered_tts3/original/`: FireRedTTS3 original weights.
- `models/reuse/original/`: RE-USE / SEMamba weights (NSCLv1, non-commercial).
