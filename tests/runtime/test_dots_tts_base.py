from __future__ import annotations

import gc
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx_speech.generation.dots_tts import DotsTTSGenerator


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("variant", ["soar", "mf"])
def test_base_checkpoint_generates_finite_non_silent_waveform(variant: str) -> None:
    model_dir = ROOT / "models/dots_tts" / variant / "mlx-base"
    if not (model_dir / "core.safetensors").is_file():
        pytest.skip(f"local dots.tts {variant} base checkpoint is unavailable")
    generator = DotsTTSGenerator.from_dir(model_dir)
    audio_vae = generator.components.audio_vae
    decoder_state = audio_vae.init_decode_state(maximum_chunk_size=16)
    assert audio_vae.decoder.input_dtype == mx.bfloat16
    assert decoder_state.decoder_input.dtype == audio_vae.decoder.input_dtype
    assert all(
        hidden.dtype == cell.dtype == mx.float32
        for hidden, cell in decoder_state.recurrent_state
    )
    result = generator.synthesize(
        "Runtime waveform check.",
        max_audio_patches=1,
        solver_steps=1,
        seed=17,
        eos_threshold=1.0,
    )
    assert result.sample_rate == 48_000
    assert result.waveform.ndim == 1
    assert result.num_patches == 1
    assert int(result.waveform.size) > 0
    assert bool(mx.all(mx.isfinite(result.waveform)).item())
    assert bool(mx.any(mx.abs(result.waveform) > 0).item())
    streamed_chunks = list(
        generator.synthesize_stream(
            "Runtime waveform check.",
            max_audio_patches=1,
            solver_steps=1,
            seed=17,
            eos_threshold=1.0,
        )
    )
    streamed = mx.concatenate([chunk.waveform for chunk in streamed_chunks])
    mx.eval(streamed)
    assert int(streamed.size) == int(result.waveform.size)
    np.testing.assert_allclose(streamed, result.waveform, atol=1e-2, rtol=1e-2)
    seams = np.cumsum([int(chunk.waveform.size) for chunk in streamed_chunks])[:-1]
    for seam in seams:
        np.testing.assert_allclose(
            streamed[int(seam) - 2 : int(seam) + 2],
            result.waveform[int(seam) - 2 : int(seam) + 2],
            atol=1e-2,
            rtol=1e-2,
        )
    repeated = generator.synthesize(
        "Runtime waveform check.",
        max_audio_patches=1,
        solver_steps=1,
        seed=17,
        eos_threshold=1.0,
    )
    np.testing.assert_array_equal(
        np.asarray(result.waveform),
        np.asarray(repeated.waveform),
    )
    del (
        audio_vae,
        decoder_state,
        generator,
        repeated,
        result,
        streamed,
        streamed_chunks,
    )
    gc.collect()
    mx.clear_cache()


@pytest.mark.parametrize("variant", ["soar", "mf"])
def test_int8_checkpoint_stream_window_uses_decoder_dtype(variant: str) -> None:
    model_dir = ROOT / "models/dots_tts" / variant / "mlx-int8"
    if not (model_dir / "core.safetensors").is_file():
        pytest.skip(f"local dots.tts {variant} int8 checkpoint is unavailable")
    generator = DotsTTSGenerator.from_dir(model_dir)
    audio_vae = generator.components.audio_vae
    state = audio_vae.init_decode_state(maximum_chunk_size=16)

    assert audio_vae.post_proj.weight.dtype == mx.bfloat16
    assert audio_vae.dec_mi_layer.input.weight.dtype == mx.bfloat16
    assert audio_vae.decoder.input_dtype == mx.bfloat16
    assert state.decoder_input.dtype == audio_vae.decoder.input_dtype
    assert all(
        hidden.dtype == cell.dtype == mx.float32
        for hidden, cell in state.recurrent_state
    )
    del audio_vae, generator, state
    gc.collect()
    mx.clear_cache()


def test_soar_base_connects_speaker_only_and_continuation_conditioning() -> None:
    model_dir = ROOT / "models/dots_tts/soar/mlx-base"
    if not (model_dir / "core.safetensors").is_file():
        pytest.skip("local dots.tts SOAR base checkpoint is unavailable")
    generator = DotsTTSGenerator.from_dir(model_dir)
    time = mx.arange(14_112, dtype=mx.float32) / 44_100.0
    reference = 0.1 * mx.sin(2.0 * np.pi * 220.0 * time)
    speaker_only = generator.synthesize(
        "Speaker-only runtime check.",
        reference_audio=reference,
        reference_sample_rate=44_100,
        max_audio_patches=1,
        solver_steps=1,
        seed=29,
        eos_threshold=1.0,
    )
    continuation = generator.synthesize(
        "Continuation runtime check.",
        reference_audio=reference,
        reference_text="Reference tone.",
        reference_sample_rate=44_100,
        max_audio_patches=3,
        solver_steps=1,
        seed=31,
        eos_threshold=1.0,
    )
    for result in (speaker_only, continuation):
        assert result.sample_rate == 48_000
        assert result.waveform.ndim == 1
        assert bool(mx.all(mx.isfinite(result.waveform)).item())
        assert bool(mx.any(mx.abs(result.waveform) > 0).item())
    assert speaker_only.num_patches == 1
    assert continuation.num_patches == 1
    del continuation, generator, reference, speaker_only
    gc.collect()
    mx.clear_cache()
