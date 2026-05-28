# Roadmap

No active roadmap.

First-time onboarding does not create roadmap phases. Refresh imports require strong roadmap evidence and user confirmation in chat.

## Deferred or Not Now

- **RE-USE reference-denoise (optional cloning-quality upgrade)** — port DramaBox's optional pre-encode speech enhancer to pure MLX so cloned-voice conditioning matches warm-server `denoise_ref=True` (cleaner result when the reference clip is noisy). **Deferred, not blocked** — it's separable from the core clone: optional (core ships `denoise_ref=False`; upstream falls back gracefully) and a self-contained model (SEMamba, ~9.6M params, Mamba/SSM-based). Implementation notes: MLX has no prebuilt fused selective-scan kernel, so the scan is hand-written — involved but well-trodden (community MLX Mamba ports exist); `nvidia/RE-USE` weights aren't local and would need fetching + conversion — check the model license before redistributing. Revisit as its own change after `2026-05-28-dramabox-voice-cloning`. Evidence: `.references/DramaBox/src/super_resolution.py`, `.references/DramaBox/src/inference_server.py:239-304`.
