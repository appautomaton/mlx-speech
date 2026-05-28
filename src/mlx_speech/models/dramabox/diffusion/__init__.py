"""DramaBox diffusion primitives — target shape, patchifier, scheduler, noiser,
state tools, and the elementary velocity/denoised conversions.

These are the moving parts of the sampling loop that do NOT depend on the
DiT itself. The DiT, guider, and full Euler loop live separately (Stage 6/7).
"""

from __future__ import annotations

from .conditioning import apply_reference_latent
from .noiser import GaussianNoiser
from .patchifier import AudioPatchifier, AudioLatentShape
from .scheduler import LTX2Scheduler
from .shape import target_shape_from_duration
from .state import AudioLatentTools, LatentState
from .utils import post_process_latent, to_denoised, to_velocity

__all__ = [
    "AudioLatentShape",
    "AudioLatentTools",
    "AudioPatchifier",
    "GaussianNoiser",
    "LTX2Scheduler",
    "LatentState",
    "apply_reference_latent",
    "post_process_latent",
    "target_shape_from_duration",
    "to_denoised",
    "to_velocity",
]
