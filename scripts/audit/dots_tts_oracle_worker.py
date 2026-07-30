#!/usr/bin/env python3
"""Isolated official-PyTorch fixture worker; never import from MLX runtime code."""

from __future__ import annotations

import argparse
import importlib.metadata
import io
import json
import platform
import random
import zipfile
from pathlib import Path

import numpy as np


SEED = 1729
MAX_ARRAY_BYTES = 16 << 20


def _numpy(tensor):
    import torch

    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.float()
        return np.ascontiguousarray(tensor.numpy())
    return np.ascontiguousarray(np.asarray(tensor))


def _save_npz(path: Path, **arrays) -> None:
    """Write deterministic, pickle-free NPZ fixtures with bounded arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {name: _numpy(value) for name, value in sorted(arrays.items())}
    for name, value in normalized.items():
        if value.dtype == object:
            raise TypeError(f"fixture array {name} cannot use object dtype")
        if value.nbytes > MAX_ARRAY_BYTES:
            raise ValueError(
                f"fixture array {name} is {value.nbytes} bytes; limit is {MAX_ARRAY_BYTES}"
            )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, value in normalized.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, value, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())


def _cache_layer_zero(cache):
    layers = getattr(cache, "layers", None)
    if layers:
        layer = layers[0]
        return layer.keys.clone(), layer.values.clone()
    layer = cache[0]
    return layer[0].clone(), layer[1].clone()


def _synthetic_audio(seconds: float = 0.64):
    import torch

    sample_rate = 48_000
    count = round(seconds * sample_rate)
    time = torch.arange(count, dtype=torch.float32) / sample_rate
    envelope = torch.linspace(0.35, 1.0, count)
    waveform = envelope * (
        0.16 * torch.sin(2 * torch.pi * 220.0 * time)
        + 0.04 * torch.sin(2 * torch.pi * 440.0 * time + 0.3)
    )
    return waveform.contiguous()


def _save_text_fixture(model, destination: Path) -> np.ndarray:
    from dots_tts.data.pipelines.tokenizing import build_generation_schedule
    from dots_tts.data.pipelines.tts_pipeline import (
        DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
        DEFAULT_TRAIN_TEMPLATE,
    )
    from dots_tts.utils.tokenizer import (
        AUDIO_COMP_SPAN_TOKEN,
        AUDIO_GEN_END_TOKEN,
        AUDIO_GEN_SPAN_TOKEN,
        AUDIO_GEN_START_TOKEN,
        TEXT_COND_END_TOKEN,
        require_token_id,
    )

    text = "[EN]Oracle fixture sentence."
    tts = build_generation_schedule(
        text=text,
        tokenizer=model.tokenizer,
        template=DEFAULT_TRAIN_TEMPLATE,
        max_audio_tokens=8,
    )
    interleave = build_generation_schedule(
        text=text,
        tokenizer=model.tokenizer,
        template=DEFAULT_INTERLEAVE_TRAIN_TEMPLATE,
        max_audio_tokens=24,
    )
    special_tokens = np.asarray(
        [
            require_token_id(model.tokenizer, token)
            for token in (
                AUDIO_GEN_START_TOKEN,
                AUDIO_GEN_SPAN_TOKEN,
                AUDIO_GEN_END_TOKEN,
                AUDIO_COMP_SPAN_TOKEN,
                TEXT_COND_END_TOKEN,
            )
        ],
        dtype=np.int64,
    )
    encoded = np.asarray(
        model.tokenizer.encode(text, add_special_tokens=False), dtype=np.int64
    )
    schedule = np.asarray(tts["schedule_ids"], dtype=np.int64)
    _save_npz(
        destination / "text_schedule.npz",
        encoded_text=encoded,
        tts_schedule=schedule,
        interleave_schedule=np.asarray(interleave["schedule_ids"], dtype=np.int64),
        special_token_ids=special_tokens,
        audio_budgets=np.asarray([8, 24], dtype=np.int64),
    )
    return schedule


def _save_qwen_fixture(core, schedule: np.ndarray, destination: Path) -> None:
    import torch

    torch.manual_seed(SEED)
    ids = torch.from_numpy(schedule[:6]).long().unsqueeze(0)
    embeddings, hidden, _logits, cache = core.step_llm(input_ids=ids)
    key_prefill, value_prefill = _cache_layer_zero(cache)
    eos_logits = core.eos_proj(hidden)

    torch.manual_seed(SEED + 1)
    next_embedding = torch.randn(1, 1, core.llm_hidden_size)
    _, hidden_decode, _decode_logits, cache = core.step_llm(
        inputs_embeds=next_embedding,
        past_key_values=cache,
    )
    key_decode, value_decode = _cache_layer_zero(cache)
    _save_npz(
        destination / "qwen.npz",
        ids=ids,
        embeddings=embeddings,
        hidden_prefill=hidden,
        eos_logits=eos_logits,
        cache_key_prefill=key_prefill,
        cache_value_prefill=value_prefill,
        next_embedding=next_embedding,
        hidden_decode=hidden_decode,
        cache_key_decode=key_decode,
        cache_value_decode=value_decode,
    )


def _save_latent_fixture(core, destination: Path) -> None:
    import torch

    latent = torch.linspace(-2.0, 2.0, 4 * core.latent_dim).reshape(
        1, 4, core.latent_dim
    )
    normalized = core.io_helper.normalize(latent)
    restored = core.io_helper.denormalize(normalized)
    _save_npz(
        destination / "latent_io.npz",
        latent=latent,
        normalized=normalized,
        restored=restored,
        mean=core.io_helper.global_mean,
        variance=core.io_helper.global_var,
    )


def _save_semantic_fixture(core, destination: Path) -> None:
    import torch

    from dots_tts.modules.backbone.encoder_inference import SemanticEncoderInference

    torch.manual_seed(SEED + 2)
    latent = torch.randn(1, 8, core.latent_dim)
    full = core.patch_encoder(latent)
    inference = SemanticEncoderInference(core.patch_encoder)
    prefill, state = inference.prefill_with_state(
        latent[:, :4], None, optimize=False
    )
    decoded, state = inference.decode_patch_with_state(
        latent[:, 4:], state, optimize=False
    )
    _save_npz(
        destination / "semantic.npz",
        latent=latent,
        full=full,
        prefill=prefill,
        decoded=decoded,
        combined=torch.cat((prefill, decoded), dim=1),
        final_sequence_length=np.asarray([state.seq_len], dtype=np.int64),
    )


def _save_speaker_fixture(model, audio, destination: Path):
    import torch

    lengths = torch.asarray([audio.numel()], dtype=torch.long)
    audio_batch = audio.unsqueeze(0)
    cropped, _original, cropped_lengths, _starts = model.xvector_extractor._crop_audio(
        audio_batch, lengths
    )
    fbank, fbank_lengths = model.xvector_extractor._extract_fbank_batch(
        cropped, cropped_lengths
    )
    embedding = model.xvector_extractor(
        audio_batch,
        audio_lengths=lengths,
        fbank=fbank,
        fbank_lengths=fbank_lengths,
    )
    projected = model.core.xvec_proj(embedding)
    _save_npz(
        destination / "speaker.npz",
        fbank=fbank[:, : int(fbank_lengths[0])],
        fbank_length=fbank_lengths,
        embedding=embedding,
        projected=projected,
    )
    return embedding, projected


def _save_audio_vae_fixture(model, audio, destination: Path) -> None:
    import torch

    encoder_audio = audio[: 8 * model.hop_size].reshape(1, 1, -1)
    distribution = model.vocoder.extract_latents(encoder_audio, do_sample=False)
    mean, log_std = torch.split(distribution, model.config.latent_dim, dim=1)
    sampled = mean.transpose(1, 2)
    normalized = model.core.io_helper.normalize(sampled)

    torch.manual_seed(SEED + 3)
    decode_latent = torch.randn(1, model.config.latent_dim, 8)
    waveform = model.vocoder.inference_from_latents(decode_latent, do_sample=False)
    _save_npz(
        destination / "audio_vae.npz",
        encoded_distribution=distribution,
        encoded_mean=mean,
        encoded_log_std=log_std,
        normalized_mean=normalized,
        decode_latent=decode_latent,
        decoded_waveform=waveform,
    )


def _save_dit_and_solver_fixture(core, g_cond, destination: Path) -> None:
    import torch

    length = 8
    patch_size = core.latent_patch_size
    torch.manual_seed(SEED + 4)
    sequence = torch.randn(1, length, core.fm_hidden_size)
    cfg_sequence = torch.randn(1, length, core.fm_hidden_size)
    mask = torch.tril(torch.ones(length, length, dtype=torch.bool)).unsqueeze(0)
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(0)
    timestep = torch.asarray([0.25], dtype=torch.float32)
    duration = torch.asarray([0.25], dtype=torch.float32)
    dit_output = core.velocity_field_predictor(
        x=sequence,
        timesteps=timestep,
        duration=duration if core.mode == "meanflow" else None,
        attn_mask=mask,
        pos_ids=positions,
        g_cond=g_cond,
    )
    _save_npz(
        destination / "dit.npz",
        sequence=sequence,
        mask=mask,
        positions=positions,
        timestep=timestep,
        duration=duration,
        g_cond=g_cond,
        output=dit_output,
    )

    torch.manual_seed(SEED + 5)
    noise = torch.randn(1, patch_size, core.latent_dim)
    z = noise.clone()
    if core.mode == "meanflow":
        steps = 4
        for step in range(steps):
            t = torch.asarray([step / steps], dtype=sequence.dtype)
            dt = torch.asarray([1.0 / steps], dtype=sequence.dtype)
            z = core.meanflow_solver_step(
                z,
                t=t,
                dt=dt,
                input_sequence=sequence,
                attn_mask=mask,
                pos_ids=positions,
                patch_size=patch_size,
                g_cond=g_cond,
            ).clone()
        guidance_scale = np.asarray([0.0], dtype=np.float32)
    else:
        steps = 2
        guidance = sequence.new_tensor(1.2)
        for step in range(steps):
            t = sequence.new_tensor([step / steps])
            velocity = core.fm_solver_step(
                t,
                z,
                input_sequence=sequence,
                cfg_sequence=cfg_sequence,
                attn_mask=mask,
                pos_ids=positions,
                hidden_size=core.hidden_patch_size,
                patch_size=patch_size,
                g_cond=g_cond,
                guidance_scale=guidance,
            )
            z = z + velocity / steps
        guidance_scale = np.asarray([1.2], dtype=np.float32)
    _save_npz(
        destination / "solver.npz",
        sequence=sequence,
        cfg_sequence=cfg_sequence,
        mask=mask,
        positions=positions,
        g_cond=g_cond,
        noise=noise,
        result=z,
        steps=np.asarray([steps], dtype=np.int64),
        guidance_scale=guidance_scale,
    )


def generate(src: Path, variant: str, destination: Path) -> None:
    import torch

    from dots_tts.models.dots_tts.model import DotsTtsModel

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_grad_enabled(False)

    model = DotsTtsModel.from_pretrained(src).float().eval()
    destination.mkdir(parents=True, exist_ok=True)
    schedule = _save_text_fixture(model, destination)
    _save_qwen_fixture(model.core, schedule, destination)
    _save_latent_fixture(model.core, destination)
    _save_semantic_fixture(model.core, destination)
    audio = _synthetic_audio()
    _embedding, g_cond = _save_speaker_fixture(model, audio, destination)
    _save_audio_vae_fixture(model, audio, destination)
    _save_dit_and_solver_fixture(model.core, g_cond, destination)

    dependencies = {}
    for package in (
        "torch",
        "torchaudio",
        "transformers",
        "torchdiffeq",
        "safetensors",
        "numpy",
        "pydantic",
        "einops",
        "loguru",
        "librosa",
        "soundfile",
    ):
        dependencies[package] = importlib.metadata.version(package)
    (destination / "worker_metadata.json").write_text(
        json.dumps(
            {
                "variant": variant,
                "python": platform.python_version(),
                "dependencies": dependencies,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--variant", choices=("soar", "mf"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.src.resolve(), args.variant, args.output.resolve())


if __name__ == "__main__":
    main()
