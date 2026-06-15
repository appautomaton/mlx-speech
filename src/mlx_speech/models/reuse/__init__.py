"""RE-USE / SEMamba speech enhancement (pure-MLX port).

Used by DramaBox `denoise_ref=True` to clean a voice reference before VAE
conditioning. Reference source: `.references/RE-USE/` (nvidia/RE-USE, NSCLv1
non-commercial weights). The selective-scan math mirrors
`.references/mamba_ssm/selective_scan_interface.py:selective_scan_ref`.
"""
