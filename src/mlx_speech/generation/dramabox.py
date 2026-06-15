"""DramaBox public wrapper — `DramaBoxModel.from_dir(...).generate(prompt)`.

Wires together every Stage 2-7 component into a single end-to-end generator:

    text prompt
      → LTXVGemmaTokenizer
      → Gemma 3 12B IT (49 hidden states)
      → FeatureExtractorV2 + Embeddings1DConnector → a_ctx [B, 1024, 2048]

    target shape from duration
      → patchifier → AudioPatchifier
      → optional voice_ref → mel → AudioVAE.encode → appended ref tokens
      → LTX2Scheduler sigmas
      → GaussianNoiser → noised state
      → X0Model + GuiderParams in euler_denoising_loop → denoised latent
      → unpatchify → silence_prior_fix → AudioVAE.decode → mel
      → VocoderWithBWE (fp32) → 48 kHz stereo waveform

Guidance this build ships with:
- ``stg_scale=1.5`` by default (STG self-attn passthrough on block 29),
  matching the DramaBox warm-server reference. Set ``stg_scale=0`` for the
  faster CFG-only path. Configurable per-call.
- ``modality_scale=1.0`` (modality guidance disabled — DramaBox warm-server
  default is also 1.0).
- Voice-reference path defaults to raw references (`denoise_ref=False`). Set
  ``denoise_ref=True`` to clean the reference with the RE-USE / SEMamba
  enhancer before VAE conditioning (pure-MLX; weights resolved lazily).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx

from .._hub import resolve_reuse_path
from .reuse import REUSEEnhancer

from ..models.dramabox.audio_vae import (
    AudioProcessor,
    AudioVAE,
    AudioVAEConfig,
    load_audio_vae_weights,
    prepare_reference_audio,
)
from ..models.dramabox.diffusion import (
    AudioLatentShape,
    AudioLatentTools,
    AudioPatchifier,
    GaussianNoiser,
    LTX2Scheduler,
    apply_reference_latent,
    target_shape_from_duration,
)
from ..models.dramabox.dit import DiTConfig, LTXModel, load_dit_weights
from ..models.dramabox.prompt import (
    DramaBoxPromptEncoder,
    Embeddings1DConnector,
    EmbeddingsProcessor,
    FeatureExtractorV2,
    load_audio_components_state,
    load_connector_weights,
    load_feature_extractor_weights,
)
from ..models.dramabox.sampling import (
    GuiderParams,
    X0Model,
    auto_rescale_for_cfg,
    euler_denoising_loop,
    silence_prior_fix,
)
from ..models.dramabox.vocoder import (
    VocoderWithBWE,
    load_vocoder_with_bwe_weights,
)
from ..models.dramabox.vocoder.vocoder import VocoderArgs
from ..models.gemma3_text import LTXVGemmaTokenizer, load_gemma3_text_model


# Negative prompt used by the warm-server reference. Mirrors
# `.references/DramaBox/src/inference_server.py:53`. CFG needs the
# negative context to *name the acoustic failure modes* it should push
# away from — empty string collapses CFG to a near no-op.
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent, robotic, distorted, noise, static, "
    "muffled, unclear, unnatural, monotone"
)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #

@dataclass
class DramaBoxResult:
    """Output of `DramaBoxModel.generate`."""

    waveform: mx.array  # [2, T_samples] fp32 in [-1, 1]
    sample_rate: int    # 48000
    duration_s: float
    settings_used: dict


# --------------------------------------------------------------------------- #
# Builder helpers
# --------------------------------------------------------------------------- #

def _build_main_vocoder_args() -> VocoderArgs:
    return VocoderArgs(
        upsample_initial_channel=1536,
        upsample_rates=(5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(11, 4, 4, 4, 4, 4),
        in_channels=128,
        out_channels=2,
        activation="snakebeta",
        use_tanh_at_final=False,
        apply_final_activation=True,
        use_bias_at_final=False,
    )


def _build_bwe_vocoder_args() -> VocoderArgs:
    return VocoderArgs(
        upsample_initial_channel=512,
        upsample_rates=(6, 5, 2, 2, 2),
        upsample_kernel_sizes=(12, 11, 4, 4, 4),
        in_channels=128,
        out_channels=2,
        activation="snakebeta",
        use_tanh_at_final=False,
        apply_final_activation=False,
        use_bias_at_final=False,
    )


# --------------------------------------------------------------------------- #
# Voice-reference denoising (RE-USE)
# --------------------------------------------------------------------------- #

def _denoise_reference_waveform(
    enhancer: REUSEEnhancer, waveform: mx.array, sample_rate: int
) -> mx.array:
    """Clean a ``[1, C, samples]`` reference waveform with RE-USE.

    Mirrors the warm-server ``_denoise_voice_ref`` shape handling
    (`.references/DramaBox/src/inference_server.py:283-300`): collapse to mono
    before the enhancer (voice cloning treats the speaker as a mono source),
    run RE-USE at the input rate (denoise-only, no resample), then re-expand to
    the original ``[1, C, samples]`` shape ``waveform_to_mel`` consumes.
    """
    if waveform.ndim != 3:
        raise ValueError(
            f"reference waveform must be [1, C, samples]; got {waveform.shape}"
        )
    batch, channels, _ = waveform.shape
    mono = waveform[0].mean(axis=0)  # [C, samples] -> [samples]
    cleaned = enhancer.enhance(mono, sample_rate)  # [samples]
    # Re-broadcast the cleaned mono signal back across the original channels.
    expanded = mx.broadcast_to(cleaned[None, None, :], (batch, channels, cleaned.shape[-1]))
    return expanded.astype(waveform.dtype)


# --------------------------------------------------------------------------- #
# DramaBoxModel
# --------------------------------------------------------------------------- #

class DramaBoxModel:
    """Loaded DramaBox runtime, ready to call ``.generate(prompt)``."""

    def __init__(
        self,
        *,
        prompt_encoder: DramaBoxPromptEncoder,
        dit: LTXModel,
        audio_vae: AudioVAE,
        vocoder: VocoderWithBWE,
        negative_a_ctx: mx.array | None = None,
        reuse_path_or_repo: str | Path | None = None,
    ):
        self.prompt_encoder = prompt_encoder
        self.dit = dit
        self.audio_vae = audio_vae
        self.vocoder = vocoder
        # Cached "empty prompt" a_ctx for CFG's uncond pass. Computed lazily
        # the first time we need it (or supplied at construction).
        self._negative_a_ctx = negative_a_ctx
        # RE-USE enhancer for `denoise_ref=True`. Weights location (None ->
        # resolve_reuse_path default) and enhancer instance are both resolved
        # lazily on the first denoise_ref=True call, then cached here.
        self._reuse_path_or_repo = reuse_path_or_repo
        self._reuse_enhancer: REUSEEnhancer | None = None
        # Cleaned references cached per (path, sample_rate) so repeated calls
        # with the same voice_ref don't re-denoise. Mirrors the warm-server
        # `_ref_denoise_cache`.
        self._ref_denoise_cache: dict[tuple[str, int], mx.array] = {}

    # ----- RE-USE enhancer (lazy) ------------------------------------------

    def _get_reuse_enhancer(self) -> REUSEEnhancer:
        """Lazily load and cache the RE-USE enhancer.

        Raises a clear, actionable error if the weights or module cannot be
        loaded, naming RE-USE and the ``denoise_ref=False`` opt-out. We never
        silently skip: a caller asking for ``denoise_ref=True`` must get either
        a cleaned reference or an explicit failure.
        """
        if self._reuse_enhancer is not None:
            return self._reuse_enhancer
        try:
            weights_dir = resolve_reuse_path(
                str(self._reuse_path_or_repo) if self._reuse_path_or_repo is not None else None
            )
            self._reuse_enhancer = REUSEEnhancer.from_dir(weights_dir)
        except Exception as exc:  # noqa: BLE001 — surface any load failure clearly
            raise RuntimeError(
                "denoise_ref=True requires the RE-USE / SEMamba enhancer, but it "
                f"could not be loaded ({type(exc).__name__}: {exc}). Provide the "
                "converted MLX weights via `reuse_path_or_repo` (or let it resolve "
                "the published repo), or set `denoise_ref=False` to use the raw "
                "voice reference."
            ) from exc
        return self._reuse_enhancer

    # ----- construction ----------------------------------------------------

    @classmethod
    def from_dir(
        cls,
        dramabox_dir: str | Path,
        *,
        gemma_dir: str | Path,
        reuse_path_or_repo: str | Path | None = None,
    ) -> "DramaBoxModel":
        """Load DramaBox from the expected directory layout.

        Args:
            dramabox_dir: path containing `dramabox-dit-v1.safetensors` and
                `dramabox-audio-components.safetensors`.
            gemma_dir: path to the pure-MLX 4-bit Gemma 3 12B IT directory.
            reuse_path_or_repo: optional location (local dir or HF repo id) of
                the converted RE-USE / SEMamba MLX weights used when
                ``denoise_ref=True``. ``None`` resolves the published repo on
                first use. Loaded lazily; ignored unless ``denoise_ref=True``.
        """
        dramabox_dir = Path(dramabox_dir)
        gemma_dir = Path(gemma_dir)

        # ----- Gemma backbone + tokenizer -----
        gemma, _ = load_gemma3_text_model(gemma_dir)
        tokenizer = LTXVGemmaTokenizer.from_dir(gemma_dir)

        # ----- Prompt pipeline -----
        audio_components = dramabox_dir / "dramabox-audio-components.safetensors"
        state = load_audio_components_state(audio_components)

        fx = FeatureExtractorV2(embedding_dim=3840, out_features=2048, num_layers=49)
        load_feature_extractor_weights(fx, state)
        connector = Embeddings1DConnector(
            num_attention_heads=32, attention_head_dim=64, num_layers=8,
            num_learnable_registers=128, positional_embedding_max_pos=4096,
            seq_len=1024,
        )
        load_connector_weights(connector, state)
        processor = EmbeddingsProcessor(feature_extractor=fx, audio_connector=connector)
        prompt_encoder = DramaBoxPromptEncoder(gemma=gemma, tokenizer=tokenizer, processor=processor)

        # ----- DiT -----
        dit_path = dramabox_dir / "dramabox-dit-v1.safetensors"
        dit = LTXModel(DiTConfig())
        dit_state = mx.load(str(dit_path))
        load_dit_weights(dit, dit_state)

        # ----- Audio VAE -----
        vae = AudioVAE(AudioVAEConfig())
        load_audio_vae_weights(vae, state)

        # ----- Vocoder + BWE -----
        vocoder = VocoderWithBWE(
            main_args=_build_main_vocoder_args(),
            bwe_args=_build_bwe_vocoder_args(),
            input_sampling_rate=16_000,
            output_sampling_rate=48_000,
            hop_length=80,
            n_fft=512,
            win_length=512,
            n_mel_channels=64,
        )
        load_vocoder_with_bwe_weights(vocoder, state)

        return cls(
            prompt_encoder=prompt_encoder,
            dit=dit,
            audio_vae=vae,
            vocoder=vocoder,
            reuse_path_or_repo=reuse_path_or_repo,
        )

    # ----- generation ------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        duration_s: float = 5.0,
        cfg_scale: float = 2.5,
        stg_scale: float = 1.5,  # warm-server default (STG on, block 29)
        rescale_scale: float | str = "auto",
        modality_scale: float = 1.0,
        steps: int = 30,
        seed: int = 42,
        voice_ref: str | Path | None = None,
        denoise_ref: bool = False,
    ) -> DramaBoxResult:
        """Generate one stereo waveform clip from a text prompt.

        When ``denoise_ref=True`` and ``voice_ref`` is set, the reference is
        collapsed to mono, cleaned with the RE-USE / SEMamba enhancer, and
        re-expanded before VAE conditioning. The diffusion path (reference
        latent application, per-token sigma) is unchanged. Default is
        ``denoise_ref=False`` (raw reference).
        """
        # Resolve rescale_scale
        rescale_val = auto_rescale_for_cfg(cfg_scale) if rescale_scale == "auto" else float(rescale_scale)

        params = GuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=rescale_val,
            modality_scale=modality_scale,
        )

        # ----- Prompt encoding -----
        a_ctx = self.prompt_encoder.encode(prompt, max_length=1024).a_ctx
        if params.needs_uncond:
            if self._negative_a_ctx is None:
                self._negative_a_ctx = self.prompt_encoder.encode(
                    DEFAULT_NEGATIVE_PROMPT, max_length=1024,
                ).a_ctx
            a_ctx_neg = self._negative_a_ctx
        else:
            a_ctx_neg = None

        # ----- Target latent shape + state -----
        target = target_shape_from_duration(duration_s)
        shape = AudioLatentShape(
            batch=target.batch,
            channels=target.channels,
            frames=target.frames,
            mel_bins=target.mel_bins,
        )
        patchifier = AudioPatchifier()
        tools = AudioLatentTools(patchifier=patchifier, target_shape=shape)
        state = tools.create_initial_state(dtype=mx.bfloat16)

        if voice_ref is not None:
            ref_audio = prepare_reference_audio(voice_ref)
            ref_waveform = ref_audio.waveform
            if denoise_ref:
                # Clean the reference between prep and the mel front end. The
                # diffusion path below is untouched. Cache per (path, sr) so a
                # repeated voice_ref does not re-denoise.
                cache_key = (str(voice_ref), int(ref_audio.sample_rate))
                cleaned = self._ref_denoise_cache.get(cache_key)
                if cleaned is None:
                    enhancer = self._get_reuse_enhancer()
                    cleaned = _denoise_reference_waveform(
                        enhancer, ref_waveform, ref_audio.sample_rate
                    )
                    mx.eval(cleaned)
                    self._ref_denoise_cache[cache_key] = cleaned
                ref_waveform = cleaned
            ref_mel = AudioProcessor().waveform_to_mel(ref_waveform, sample_rate=ref_audio.sample_rate)
            ref_latent = self.audio_vae.encode(ref_mel)
            state = apply_reference_latent(state, ref_latent, patchifier=patchifier)
            # Per-token timesteps (denoise_mask * sigma) keep the appended reference
            # tokens modulated as clean (timestep 0) in the DiT AdaLN. None for the
            # no-ref path → bit-identical scalar-sigma baseline.
            loop_denoise_mask = state.denoise_mask
        else:
            loop_denoise_mask = None

        # ----- Noise -----
        noiser = GaussianNoiser(seed=seed)
        state = noiser(state, noise_scale=1.0)

        # ----- Sigma schedule -----
        sigmas = LTX2Scheduler().execute(steps=steps, tokens=state.latent.shape[-1])

        # ----- Denoising loop -----
        # Pass the patchifier's [B, 1, T, 2] start/end timings so the DiT's
        # RoPE matches the reference (max_pos=20, use_middle_indices_grid=True).
        x0 = X0Model(self.dit)
        state = euler_denoising_loop(
            state,
            sigmas,
            x0_model=x0,
            a_ctx=a_ctx,
            a_ctx_neg=a_ctx_neg,
            params=params,
            positions=state.positions,
            denoise_mask=loop_denoise_mask,
        )

        # ----- Strip appended reference tokens before VAE decode -----
        state = tools.clear_conditioning(state)

        # ----- Unpatchify, silence-prior fix -----
        state = tools.unpatchify_state(state)
        latent_4d = silence_prior_fix(state.latent)

        # ----- VAE decode -----
        mel = self.audio_vae.decode(latent_4d)  # [B, 2, T_mel, 64]

        # ----- Vocoder + BWE -----
        waveform = self.vocoder(mel)  # [B, 2, T_samples] at 48 kHz

        return DramaBoxResult(
            waveform=waveform[0].astype(mx.float32),
            sample_rate=48_000,
            duration_s=duration_s,
            settings_used={
                "cfg_scale": cfg_scale,
                "stg_scale": stg_scale,
                "rescale_scale": rescale_val,
                "modality_scale": modality_scale,
                "steps": steps,
                "seed": seed,
                "duration_s": duration_s,
                "voice_ref": str(voice_ref) if voice_ref is not None else None,
                "denoise_ref": denoise_ref,
            },
        )


__all__ = ["DramaBoxModel", "DramaBoxResult"]
