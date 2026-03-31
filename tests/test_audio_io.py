import mlx.core as mx

from mlx_speech.audio import normalize_peak, trim_leading_silence


def test_trim_leading_silence_removes_low_energy_prefix() -> None:
    silence = mx.zeros((2400,), dtype=mx.float32)
    tone = mx.full((2400,), 0.1, dtype=mx.float32)
    waveform = mx.concatenate([silence, tone], axis=0)

    trimmed = trim_leading_silence(
        waveform,
        sample_rate=24000,
        threshold=0.01,
        frame_ms=20.0,
        keep_ms=0.0,
    )

    assert trimmed.shape[0] < waveform.shape[0]
    assert abs(float(trimmed[0].item()) - 0.1) < 1e-6


def test_normalize_peak_increases_peak_up_to_target() -> None:
    waveform = mx.array([0.0, 0.25, -0.5, 0.1], dtype=mx.float32)
    normalized = normalize_peak(waveform, target_peak=1.0, max_gain=4.0)

    assert abs(float(mx.max(mx.abs(normalized)).item()) - 1.0) < 1e-6
