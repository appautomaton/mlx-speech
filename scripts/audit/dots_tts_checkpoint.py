#!/usr/bin/env python3
"""Audit strict base loading and official-fixture parity for dots.tts artifacts."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_speech.models.dots_tts.audio_vae import (
    AudioVAE,
    encoder_logical_workspace_bytes,
)
from mlx_speech.models.dots_tts.checkpoint import (
    _strict_load,
    load_dots_tts_components,
    storage_dtype,
    validate_artifact_dir,
)
from mlx_speech.models.dots_tts.solvers import MeanFlowSolver, SOARSolver
from mlx_speech.models.dots_tts.speaker import SpeakerFrontend


@dataclass(frozen=True)
class TensorMetric:
    name: str
    max_absolute_error: float
    cosine_similarity: float
    atol: float
    rtol: float


@dataclass(frozen=True)
class VariantAudit:
    variant: str
    artifact_dir: str
    strict_components: tuple[str, ...]
    metrics: tuple[TensorMetric, ...]


@dataclass(frozen=True)
class EncoderBenchmark:
    variant: str
    seconds: float
    samples: int
    latent_frames: int
    duration_seconds: float
    baseline_active_bytes: int
    peak_active_bytes: int
    incremental_peak_bytes: int
    logical_workspace_bound_bytes: int


def _array(value: mx.array | np.ndarray) -> np.ndarray:
    if isinstance(value, mx.array):
        value = value.astype(mx.float32)
    return np.asarray(value, dtype=np.float32)


def _metric(
    name: str,
    actual: mx.array | np.ndarray,
    expected: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> TensorMetric:
    actual_array = _array(actual)
    expected_array = np.asarray(expected, dtype=np.float32)
    if actual_array.shape != expected_array.shape:
        raise AssertionError(
            f"{name} shape mismatch: {actual_array.shape} != {expected_array.shape}"
        )
    difference = np.abs(actual_array - expected_array)
    maximum = float(difference.max(initial=0.0))
    actual_flat = actual_array.reshape(-1).astype(np.float64)
    expected_flat = expected_array.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(actual_flat) * np.linalg.norm(expected_flat)
    cosine = (
        1.0
        if denominator == 0.0 and np.array_equal(actual_array, expected_array)
        else float(np.dot(actual_flat, expected_flat) / max(denominator, 1e-30))
    )
    if not np.allclose(actual_array, expected_array, atol=atol, rtol=rtol):
        raise AssertionError(
            f"{name} parity failed: max_abs={maximum:.7g}, cosine={cosine:.9f}, "
            f"atol={atol}, rtol={rtol}"
        )
    return TensorMetric(name, maximum, cosine, atol, rtol)


def _synthetic_audio(seconds: float = 0.64) -> np.ndarray:
    sample_rate = 48_000
    count = round(seconds * sample_rate)
    time = np.arange(count, dtype=np.float32) / sample_rate
    envelope = np.linspace(0.35, 1.0, count, dtype=np.float32)
    return envelope * (
        0.16 * np.sin(2 * np.pi * 220.0 * time)
        + 0.04 * np.sin(2 * np.pi * 440.0 * time + 0.3)
    )


def _fixture(root: Path, variant: str, name: str) -> dict[str, np.ndarray]:
    with np.load(root / variant / name, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def audit_variant(
    artifact_dir: str | Path,
    *,
    variant: str,
    fixture_root: str | Path = "tests/fixtures/dots_tts",
) -> VariantAudit:
    loaded = load_dots_tts_components(artifact_dir)
    if loaded.layout.artifact_config.variant != variant:
        raise ValueError("artifact variant does not match requested audit")
    core = loaded.core
    if not all(report.is_exact_match for report in loaded.reports):
        raise AssertionError("artifact storage/runtime dtype validation did not pass")
    # The official worker materializes the source BF16 core checkpoint with FP32
    # compute. Strict loading above independently validates both stored and loaded
    # parameter dtypes before this parity-only cast. Speaker and AudioVAE stay at
    # their released mixed runtime dtypes, so decoder BF16 regressions remain visible.
    core.set_dtype(mx.float32)
    fixtures = Path(fixture_root)
    metrics: list[TensorMetric] = []

    latent = _fixture(fixtures, variant, "latent_io.npz")
    normalized = loaded.latent_io.normalize(mx.array(latent["latent"]))
    restored = loaded.latent_io.denormalize(normalized)
    metrics.extend(
        (
            _metric("latent.normalized", normalized, latent["normalized"], atol=1e-5, rtol=1e-5),
            _metric("latent.restored", restored, latent["restored"], atol=1e-5, rtol=1e-5),
        )
    )

    qwen = _fixture(fixtures, variant, "qwen.npz")
    ids = mx.array(qwen["ids"], dtype=mx.int32)
    embeddings = core.qwen.get_input_embeddings()(ids)
    prefill = core.qwen(input_ids=ids, request_logits=False)
    first_key, first_value = prefill.cache[0]
    decoded = core.qwen.step(
        inputs_embeds=mx.array(qwen["next_embedding"]),
        cache=prefill.cache,
        request_logits=False,
    )
    decoded_key, decoded_value = decoded.cache[0]
    metrics.extend(
        (
            _metric("qwen.embeddings", embeddings, qwen["embeddings"], atol=0.01, rtol=0.01),
            _metric("qwen.hidden_prefill", prefill.last_hidden_state, qwen["hidden_prefill"], atol=0.01, rtol=0.01),
            _metric("qwen.eos_logits", prefill.eos_logits, qwen["eos_logits"], atol=0.01, rtol=0.01),
            _metric("qwen.cache_key_prefill", first_key.transpose(0, 2, 1, 3), qwen["cache_key_prefill"], atol=0.01, rtol=0.01),
            _metric("qwen.cache_value_prefill", first_value.transpose(0, 2, 1, 3), qwen["cache_value_prefill"], atol=0.01, rtol=0.01),
            _metric("qwen.hidden_decode", decoded.last_hidden_state, qwen["hidden_decode"], atol=0.01, rtol=0.01),
            _metric("qwen.cache_key_decode", decoded_key.transpose(0, 2, 1, 3), qwen["cache_key_decode"], atol=0.01, rtol=0.01),
            _metric("qwen.cache_value_decode", decoded_value.transpose(0, 2, 1, 3), qwen["cache_value_decode"], atol=0.01, rtol=0.01),
        )
    )

    semantic = _fixture(fixtures, variant, "semantic.npz")
    semantic_latent = mx.array(semantic["latent"])
    semantic_full = core.semantic_encoder(semantic_latent)
    semantic_prefill, semantic_state = core.semantic_encoder.prefill(
        semantic_latent[:, :4]
    )
    semantic_decode, semantic_state = core.semantic_encoder.decode_patch(
        semantic_latent[:, 4:], semantic_state
    )
    metrics.extend(
        (
            _metric("semantic.full", semantic_full, semantic["full"], atol=0.02, rtol=0.02),
            _metric("semantic.prefill", semantic_prefill, semantic["prefill"], atol=0.02, rtol=0.02),
            _metric("semantic.decode", semantic_decode, semantic["decoded"], atol=0.02, rtol=0.02),
        )
    )
    if semantic_state.sequence_length != int(semantic["final_sequence_length"][0]):
        raise AssertionError("semantic final sequence length differs from oracle")

    speaker = _fixture(fixtures, variant, "speaker.npz")
    features, feature_length = SpeakerFrontend().features(
        _synthetic_audio(), sample_rate=48_000
    )
    if feature_length != int(speaker["fbank_length"][0]):
        raise AssertionError("speaker fbank length differs from oracle")
    embedding = loaded.speaker_encoder(
        mx.array(speaker["fbank"]),
        lengths=mx.array(speaker["fbank_length"], dtype=mx.int32),
    )
    projected = core.speaker_projection_norm(core.speaker_projection(embedding))
    metrics.extend(
        (
            _metric("speaker.fbank", features[None], speaker["fbank"], atol=0.01, rtol=0.01),
            _metric("speaker.embedding", embedding, speaker["embedding"], atol=0.01, rtol=0.01),
            _metric("speaker.projected", projected, speaker["projected"], atol=0.01, rtol=0.01),
        )
    )

    audio = _fixture(fixtures, variant, "audio_vae.npz")
    encoder_audio = mx.array(_synthetic_audio()[: 8 * loaded.audio_vae.hop_size])[None, None]
    distribution = loaded.audio_vae.encode(encoder_audio)
    mean, log_standard_deviation = mx.split(distribution, 2, axis=1)
    normalized_mean = loaded.latent_io.normalize(mean.transpose(0, 2, 1))
    decoded_waveform = loaded.audio_vae.decode(mx.array(audio["decode_latent"]))
    metrics.extend(
        (
            _metric("audio.distribution", distribution, audio["encoded_distribution"], atol=0.02, rtol=0.02),
            _metric("audio.mean", mean, audio["encoded_mean"], atol=0.02, rtol=0.02),
            _metric("audio.log_std", log_standard_deviation, audio["encoded_log_std"], atol=0.02, rtol=0.02),
            _metric("audio.normalized_mean", normalized_mean, audio["normalized_mean"], atol=0.02, rtol=0.02),
            _metric("audio.waveform", decoded_waveform, audio["decoded_waveform"], atol=0.02, rtol=0.02),
        )
    )

    dit = _fixture(fixtures, variant, "dit.npz")
    dit_output = core.dit(
        mx.array(dit["sequence"]),
        mx.array(dit["timestep"]),
        duration=(mx.array(dit["duration"]) if variant == "mf" else None),
        attention_mask=mx.array(dit["mask"]),
        positions=mx.array(dit["positions"]),
        speaker_condition=mx.array(dit["g_cond"]),
    )
    metrics.append(
        _metric("dit.output", dit_output, dit["output"], atol=0.02, rtol=0.02)
    )

    solver = _fixture(fixtures, variant, "solver.npz")
    common = {
        "sequence": mx.array(solver["sequence"]),
        "attention_mask": mx.array(solver["mask"]),
        "positions": mx.array(solver["positions"]),
        "speaker_condition": mx.array(solver["g_cond"]),
        "steps": int(solver["steps"][0]),
        "patch_size": int(solver["noise"].shape[1]),
        "noise": mx.array(solver["noise"]),
    }
    if variant == "mf":
        solver_model = MeanFlowSolver(
            core.dit, core.coordinate_projection, latent_dim=loaded.layout.config.latent_dim
        )
        solver_result = solver_model.sample(**common)
    else:
        solver_model = SOARSolver(
            core.dit, core.coordinate_projection, latent_dim=loaded.layout.config.latent_dim
        )
        solver_result = solver_model.sample(
            **common,
            cfg_sequence=mx.array(solver["cfg_sequence"]),
            guidance_scale=float(solver["guidance_scale"][0]),
        )
    metrics.append(
        _metric("solver.result", solver_result, solver["result"], atol=0.03, rtol=0.03)
    )
    return VariantAudit(
        variant=variant,
        artifact_dir=str(artifact_dir),
        strict_components=tuple(report.component for report in loaded.reports),
        metrics=tuple(metrics),
    )


def benchmark_encoder(
    artifact_dir: str | Path,
    *,
    variant: str,
    seconds: float,
    logical_limit_bytes: int,
    metal_limit_bytes: int,
) -> EncoderBenchmark:
    """Measure a representative official FP32 encode without decoder changes."""

    if seconds <= 0:
        raise ValueError("encoder benchmark seconds must be positive")
    layout = validate_artifact_dir(artifact_dir)
    audio_vae = AudioVAE(layout.config.vocoder)
    policy = layout.artifact_config.dtype_policy
    _strict_load(
        "vocoder",
        audio_vae,
        layout.model_dir / "vocoder.safetensors",
        expected_dtype=lambda path: storage_dtype(policy, "vocoder", path),
    )
    sample_count = round(seconds * layout.config.vocoder.sample_rate)
    logical_bound = encoder_logical_workspace_bytes(
        layout.config.vocoder, sample_count=sample_count
    )
    if logical_bound > logical_limit_bytes:
        raise AssertionError(
            f"encoder logical workspace {logical_bound} exceeds {logical_limit_bytes}"
        )

    gc.collect()
    mx.clear_cache()
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    waveform = mx.array(_synthetic_audio(seconds))[None, None]
    started = time.perf_counter()
    distribution = audio_vae.encode(waveform)
    mx.eval(distribution)
    duration = time.perf_counter() - started
    peak = int(mx.get_peak_memory())
    incremental_peak = max(0, peak - baseline)
    if incremental_peak > metal_limit_bytes:
        raise AssertionError(
            f"encoder Metal peak {incremental_peak} exceeds {metal_limit_bytes}"
        )
    return EncoderBenchmark(
        variant=variant,
        seconds=seconds,
        samples=sample_count,
        latent_frames=int(distribution.shape[-1]),
        duration_seconds=duration,
        baseline_active_bytes=baseline,
        peak_active_bytes=peak,
        incremental_peak_bytes=incremental_peak,
        logical_workspace_bound_bytes=logical_bound,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("soar", "mf", "all"), default="all")
    parser.add_argument("--precision", choices=("base",), default="base")
    parser.add_argument("--root", type=Path, default=Path("models/dots_tts"))
    parser.add_argument("--benchmark-encoder-seconds", type=float)
    parser.add_argument("--encoder-logical-limit-mib", type=float, default=32.0)
    parser.add_argument("--encoder-metal-limit-mib", type=float, default=1792.0)
    args = parser.parse_args()
    variants = ("soar", "mf") if args.variant == "all" else (args.variant,)
    audits = []
    for variant in variants:
        audits.append(
            audit_variant(
                args.root / variant / f"mlx-{args.precision}", variant=variant
            )
        )
        gc.collect()
        mx.clear_cache()
    if args.benchmark_encoder_seconds is None:
        payload: object = [asdict(audit) for audit in audits]
    else:
        benchmarks = []
        for variant in variants:
            benchmarks.append(
                benchmark_encoder(
                    args.root / variant / f"mlx-{args.precision}",
                    variant=variant,
                    seconds=args.benchmark_encoder_seconds,
                    logical_limit_bytes=round(
                        args.encoder_logical_limit_mib * 1024 * 1024
                    ),
                    metal_limit_bytes=round(args.encoder_metal_limit_mib * 1024 * 1024),
                )
            )
            gc.collect()
            mx.clear_cache()
        payload = {
            "audits": [asdict(audit) for audit in audits],
            "encoder_benchmarks": [
                asdict(benchmark) for benchmark in benchmarks
            ],
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
