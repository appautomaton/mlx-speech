"""Generation loop for VibeVoice Large.

Autoregressive token generation interleaved with per-frame diffusion decoding
and semantic feedback.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import re

from ..models.vibevoice.acoustic import VibeVoiceConvCache
from ..models.vibevoice.model import VibeVoiceForConditionalGeneration
from ..models.vibevoice.tokenizer import VibeVoiceTokenizer


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VibeVoiceGenerationConfig:
    max_new_tokens: int = 4096
    cfg_scale: float = 1.3
    diffusion_steps: int = 20
    diffusion_steps_fast: int | None = None
    diffusion_warmup_frames: int = 10
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float | None = 1.0
    seed: int | None = None
    safety_max_new_tokens: int = 8192


# --------------------------------------------------------------------------- #
# Output types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class VibeVoiceGenerationOutput:
    """Raw generation output before audio concatenation."""

    audio_chunks: tuple[mx.array, ...]  # per-chunk waveform arrays
    generated_tokens: int
    stop_reached: bool


@dataclass(frozen=True)
class VibeVoiceSynthesisOutput:
    """Final synthesis output with concatenated waveform."""

    waveform: mx.array  # (T_audio,) float32 mono
    sample_rate: int
    generated_tokens: int
    stop_reached: bool


# --------------------------------------------------------------------------- #
# Token constraint
# --------------------------------------------------------------------------- #

def _constrain_logits(logits: mx.array, valid_ids: list[int]) -> mx.array:
    """Mask all logits to -inf except the given valid token IDs."""
    mask = mx.full(logits.shape, float("-inf"), dtype=mx.float32)
    for tid in valid_ids:
        mask[:, tid] = 0.0
    return logits.astype(mx.float32) + mask


def _apply_top_p(logits: mx.array, top_p: float | None) -> mx.array:
    if top_p is None or top_p >= 1.0:
        return logits
    if top_p <= 0.0:
        raise ValueError("top_p must be > 0 when provided.")

    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = mx.softmax(sorted_logits, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    to_remove = cumulative_probs > top_p
    to_remove = mx.concatenate(
        [
            mx.zeros_like(to_remove[..., :1]),
            to_remove[..., :-1],
        ],
        axis=-1,
    )
    neg_inf = mx.array(mx.finfo(logits.dtype).min, dtype=logits.dtype)
    filtered_sorted_logits = mx.where(to_remove, neg_inf, sorted_logits)
    filtered_logits = mx.full(logits.shape, neg_inf, dtype=logits.dtype)
    return mx.put_along_axis(filtered_logits, sorted_indices, filtered_sorted_logits, axis=-1)


def _sample_next_token(
    logits: mx.array,
    *,
    valid_ids: list[int],
    temperature: float,
    top_p: float | None,
    do_sample: bool,
) -> mx.array:
    constrained = _constrain_logits(logits, valid_ids)
    if not do_sample or temperature <= 0.0:
        return mx.argmax(constrained, axis=-1)

    warped = constrained / temperature
    warped = _apply_top_p(warped, top_p)
    return mx.random.categorical(warped, axis=-1)


# --------------------------------------------------------------------------- #
# Prompt formatting
# --------------------------------------------------------------------------- #

_SPEAKER_LABEL_RE = re.compile(r"^\s*Speaker\s+\d+\s*:", re.IGNORECASE)
_BRACKET_SPEAKER_RE = re.compile(r"\[(\d+)\]\s*:")


def _format_text_input(text: str) -> str:
    """Format user text for the VibeVoice prompt.

    - Preserve explicit `Speaker N:` multi-speaker scripts as-is.
    - Convert `[N]:` tags to `Speaker N:`.
    - For plain text, assign it to `Speaker 1:`.
    """
    text = text.strip()
    if not text:
        raise ValueError("text must not be empty")

    if _SPEAKER_LABEL_RE.match(text):
        return text

    if _BRACKET_SPEAKER_RE.search(text):
        return _BRACKET_SPEAKER_RE.sub(lambda m: f"Speaker {int(m.group(1))}:", text)

    text = re.sub(r"\s+", " ", text)
    return f"Speaker 1: {text}"


# --------------------------------------------------------------------------- #
# Main generation function
# --------------------------------------------------------------------------- #

def generate_vibevoice(
    model: VibeVoiceForConditionalGeneration,
    tokenizer: VibeVoiceTokenizer,
    text: str,
    *,
    reference_audio: mx.array | None = None,
    voice_samples: list[mx.array] | None = None,
    config: VibeVoiceGenerationConfig | None = None,
) -> VibeVoiceGenerationOutput:
    """Generate speech from text using VibeVoice Large.

    Args:
        model: loaded VibeVoice model
        tokenizer: loaded VibeVoice tokenizer
        text: input text. For multi-speaker, use format:
            "Speaker 1: First line\\nSpeaker 2: Second line\\n..."
        reference_audio: optional (1, 1, T) single reference for voice cloning
        voice_samples: optional list of (1, 1, T) references, one per speaker
            (for multi-speaker). Overrides reference_audio if provided.
        config: generation config (defaults used if None)

    Returns:
        VibeVoiceGenerationOutput with audio chunks
    """
    if config is None:
        config = VibeVoiceGenerationConfig()
    if config.seed is not None:
        mx.random.seed(int(config.seed))

    system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
    B = 1
    valid_ids = tokenizer.valid_speech_token_ids

    # Normalize voice samples: single reference_audio → voice_samples list
    if voice_samples is None and reference_audio is not None:
        voice_samples = [reference_audio]

    if voice_samples is not None and len(voice_samples) > 0:
        # --- Voice cloning prompt ---
        # Encode all voice references
        all_ref_latents: list[mx.array] = []
        for vs in voice_samples:
            latents = model.encode_reference_audio(vs)
            mx.eval(latents)
            all_ref_latents.append(latents)

        # Build token sequence:
        # [system] " Voice input:\n"
        # " Speaker 0:" [speech_start] [placeholders] [speech_end] "\n"
        # " Speaker 1:" [speech_start] [placeholders] [speech_end] "\n"
        # ...
        # " Text input:\n" [text lines] " Speech output:\n" [speech_start]
        prompt_tokens: list[int] = []
        speech_input_mask: list[bool] = []
        # Track which reference latent each masked position maps to
        mask_to_ref: list[tuple[int, int]] = []  # (speaker_idx, frame_idx)

        # System prompt
        sys_toks = tokenizer.encode(system_prompt, add_special_tokens=False)
        prompt_tokens.extend(sys_toks)
        speech_input_mask.extend([False] * len(sys_toks))

        # Voice input section
        vi_toks = tokenizer.encode(" Voice input:\n", add_special_tokens=False)
        prompt_tokens.extend(vi_toks)
        speech_input_mask.extend([False] * len(vi_toks))

        # Per-speaker voice reference
        for spk_idx, ref_lat in enumerate(all_ref_latents):
            n_frames = ref_lat.shape[1]

            sp_toks = tokenizer.encode(f" Speaker {spk_idx}:", add_special_tokens=False)
            prompt_tokens.extend(sp_toks)
            speech_input_mask.extend([False] * len(sp_toks))

            prompt_tokens.append(tokenizer.speech_start_id)
            speech_input_mask.append(False)

            prompt_tokens.extend([tokenizer.speech_diffusion_id] * n_frames)
            speech_input_mask.extend([True] * n_frames)
            for f in range(n_frames):
                mask_to_ref.append((spk_idx, f))

            prompt_tokens.append(tokenizer.speech_end_id)
            speech_input_mask.append(False)

            nl_toks = tokenizer.encode("\n", add_special_tokens=False)
            prompt_tokens.extend(nl_toks)
            speech_input_mask.extend([False] * len(nl_toks))

        # Text input section — rewrite Speaker N labels to 0-based
        text_lines = _format_text_input(text)
        # Normalize "Speaker N:" to 0-based indexing
        def _reindex(m: re.Match) -> str:
            n = int(m.group(1))
            return f"Speaker {n - 1}:" if n > 0 else f"Speaker {n}:"
        text_0based = re.sub(r"Speaker\s+(\d+)\s*:", _reindex, text_lines)

        for part in [" Text input:\n", f" {text_0based}\n", " Speech output:\n"]:
            toks = tokenizer.encode(part, add_special_tokens=False)
            prompt_tokens.extend(toks)
            speech_input_mask.extend([False] * len(toks))

        prompt_tokens.append(tokenizer.speech_start_id)
        speech_input_mask.append(False)

        # Build embeddings and splice voice references at masked positions
        input_ids = mx.array([prompt_tokens], dtype=mx.int32)
        inputs_embeds = model.embed_tokens(input_ids)

        mask_indices = [i for i, m in enumerate(speech_input_mask) if m]
        for i, idx in enumerate(mask_indices):
            spk_idx, frame_idx = mask_to_ref[i]
            inputs_embeds[:, idx, :] = all_ref_latents[spk_idx][:, frame_idx, :]
    else:
        # No voice cloning — preserve explicit speaker labels when provided.
        text_input = _format_text_input(text)
        prompt_parts = [
            system_prompt,
            " Text input:\n",
            f" {text_input}\n",
            " Speech output:\n",
        ]
        prompt_tokens = []
        for part in prompt_parts:
            prompt_tokens.extend(tokenizer.encode(part, add_special_tokens=False))
        prompt_tokens.append(tokenizer.speech_start_id)
        input_ids = mx.array([prompt_tokens], dtype=mx.int32)
        inputs_embeds = model.embed_tokens(input_ids)

    # Prefill positive path
    logits, hidden, pos_cache = model.lm_forward(inputs_embeds=inputs_embeds)

    # Prefill negative path (just speech_start token)
    neg_start = mx.array([[tokenizer.speech_start_id]], dtype=mx.int32)
    neg_embeds = model.embed_tokens(neg_start)
    _, neg_hidden, neg_cache = model.lm_forward(inputs_embeds=neg_embeds)

    # --- Decode loop with streaming semantic feedback ---
    acoustic_cache = VibeVoiceConvCache()
    semantic_cache = VibeVoiceConvCache()
    audio_chunks: list[mx.array] = []
    generated_tokens = 0
    stop_reached = False
    max_steps = min(config.max_new_tokens, config.safety_max_new_tokens)

    # Get first token from prefill logits
    last_logits = logits[:, -1:, :]  # (1, 1, V)
    last_hidden = hidden[:, -1:, :]  # (1, 1, H)

    for step in range(max_steps):
        next_token = _sample_next_token(
            last_logits[:, -1, :],
            valid_ids=valid_ids,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
        )

        next_token_id = next_token.item()
        generated_tokens += 1

        # --- Dispatch on token ---
        if next_token_id == tokenizer.eos_token_id:
            stop_reached = True
            break

        if next_token_id == tokenizer.speech_end_id:
            # Speech segment ended — reset streaming caches
            acoustic_cache.clear()
            semantic_cache.clear()
            next_embeds = model.embed_tokens(next_token.reshape(1, 1))
            last_logits, last_hidden, pos_cache = model.lm_forward(
                inputs_embeds=next_embeds, cache=pos_cache,
            )
            _, _, neg_cache = model.lm_forward(
                inputs_embeds=next_embeds, cache=neg_cache,
            )
            continue

        if next_token_id == tokenizer.speech_start_id:
            # Reset negative cache for new speech segment
            neg_embeds = model.embed_tokens(
                mx.array([[tokenizer.speech_start_id]], dtype=mx.int32)
            )
            _, neg_hidden, neg_cache = model.lm_forward(inputs_embeds=neg_embeds)

            next_embeds = model.embed_tokens(next_token.reshape(1, 1))
            last_logits, last_hidden, pos_cache = model.lm_forward(
                inputs_embeds=next_embeds, cache=pos_cache,
            )
            continue

        if next_token_id == tokenizer.speech_diffusion_id:
            # Run negative path decode step
            neg_token_embeds = model.embed_tokens(next_token.reshape(1, 1))
            _, neg_step_hidden, neg_cache = model.lm_forward(
                inputs_embeds=neg_token_embeds, cache=neg_cache,
            )

            # Diffusion: condition on last hidden states
            pos_condition = last_hidden[:, -1, :]  # (B, H)
            neg_condition = neg_step_hidden[:, -1, :]  # (B, H)

            # Adaptive diffusion steps: full steps for warmup, fewer after
            n_frames = len(audio_chunks)
            if (
                config.diffusion_steps_fast is not None
                and n_frames >= config.diffusion_warmup_frames
            ):
                steps = config.diffusion_steps_fast
            else:
                steps = config.diffusion_steps

            speech_latent = model.sample_speech_tokens(
                pos_condition,
                neg_condition,
                cfg_scale=config.cfg_scale,
                num_steps=steps,
            )  # (B, latent_size)

            # Streaming acoustic decode: latent → audio chunk
            audio_chunk = model.decode_latent_to_audio(
                speech_latent, cache=acoustic_cache,
            )  # (B, 1, T_chunk)
            audio_chunks.append(audio_chunk[0, 0])

            # Streaming semantic encode: audio → semantic features
            semantic_features = model.encode_semantic(
                audio_chunk, cache=semantic_cache,
            )  # (B, T_frames, semantic_dim)

            # Build next embedding: acoustic + semantic connectors
            acoustic_embed = model.model.acoustic_connector(
                speech_latent.reshape(B, 1, -1)
            )  # (B, 1, H)
            semantic_embed = model.model.semantic_connector(
                semantic_features
            )  # (B, T_frames, H)
            if semantic_embed.shape[1] > 1:
                semantic_embed = semantic_embed[:, -1:, :]
            next_embeds = acoustic_embed + semantic_embed  # (B, 1, H)

            last_logits, last_hidden, pos_cache = model.lm_forward(
                inputs_embeds=next_embeds, cache=pos_cache,
            )
            continue

    return VibeVoiceGenerationOutput(
        audio_chunks=tuple(audio_chunks),
        generated_tokens=generated_tokens,
        stop_reached=stop_reached,
    )


# --------------------------------------------------------------------------- #
# High-level synthesis wrapper
# --------------------------------------------------------------------------- #

def synthesize_vibevoice(
    model: VibeVoiceForConditionalGeneration,
    tokenizer: VibeVoiceTokenizer,
    text: str,
    *,
    reference_audio: mx.array | None = None,
    voice_samples: list[mx.array] | None = None,
    config: VibeVoiceGenerationConfig | None = None,
) -> VibeVoiceSynthesisOutput:
    """Generate speech and concatenate to a single waveform.

    Args:
        model: loaded VibeVoice model
        tokenizer: loaded VibeVoice tokenizer
        text: input text. For multi-speaker use "Speaker 1: ...\\nSpeaker 2: ..."
        reference_audio: optional (1, 1, T) single reference for voice cloning
        voice_samples: optional list of (1, 1, T) references for multi-speaker
        config: generation config

    Returns:
        VibeVoiceSynthesisOutput with concatenated waveform
    """
    gen_output = generate_vibevoice(
        model, tokenizer, text,
        reference_audio=reference_audio,
        voice_samples=voice_samples,
        config=config,
    )

    if gen_output.audio_chunks:
        waveform = mx.concatenate(list(gen_output.audio_chunks))
    else:
        waveform = mx.array([], dtype=mx.float32)

    return VibeVoiceSynthesisOutput(
        waveform=waveform,
        sample_rate=model.config.sampling_rate,
        generated_tokens=gen_output.generated_tokens,
        stop_reached=gen_output.stop_reached,
    )
