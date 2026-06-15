"""Unit-level checks for DramaBox `denoise_ref` wiring (Slice 7).

No heavy weights here: these assert the public contract (default stays False,
the mono-collapse/re-expand helper, and the clear error when RE-USE is
unavailable). The end-to-end behavior is gated in the runtime test.
"""

from __future__ import annotations

import inspect

import mlx.core as mx
import pytest

from mlx_speech.generation import dramabox as dramabox_mod
from mlx_speech.generation.dramabox import (
    DramaBoxModel,
    _denoise_reference_waveform,
)


def test_denoise_ref_default_is_false():
    """The public `generate` default must stay False (opt-in cleaning)."""
    sig = inspect.signature(DramaBoxModel.generate)
    assert sig.parameters["denoise_ref"].default is False


def test_from_dir_accepts_reuse_path_or_repo():
    """`from_dir` exposes the RE-USE weights injection point."""
    sig = inspect.signature(DramaBoxModel.from_dir)
    assert "reuse_path_or_repo" in sig.parameters
    assert sig.parameters["reuse_path_or_repo"].default is None


def test_denoise_reference_waveform_mono_collapse_and_reexpand():
    """Helper collapses [1, C, T] to mono for RE-USE, then restores the shape.

    Uses a stub enhancer (identity on the mono signal) so the test needs no
    weights. The two input channels differ, so a correct mono collapse averages
    them; the output channels are then identical (re-broadcast mono).
    """

    class _StubEnhancer:
        def __init__(self):
            self.calls = []

        def enhance(self, waveform, in_sr):
            # RE-USE contract: mono-in (T,) / mono-out (T,).
            assert waveform.ndim == 1
            self.calls.append((tuple(waveform.shape), in_sr))
            return waveform

    left = mx.arange(8, dtype=mx.float32)
    right = mx.arange(8, dtype=mx.float32) + 2.0
    waveform = mx.stack([left, right], axis=0)[None]  # [1, 2, 8]

    stub = _StubEnhancer()
    out = _denoise_reference_waveform(stub, waveform, 16_000)
    mx.eval(out)

    assert out.shape == waveform.shape  # [1, 2, 8] restored
    # Enhancer saw a single mono (T,) signal at the input rate.
    assert stub.calls == [((8,), 16_000)]
    # Output channels are the re-broadcast cleaned mono (identical).
    assert mx.allclose(out[0, 0], out[0, 1]).item()
    # The mono signal is the per-sample mean of the two channels.
    expected_mono = (left + right) / 2.0
    assert mx.allclose(out[0, 0], expected_mono).item()


def test_denoise_reference_waveform_rejects_non_3d():
    """The helper guards the [1, C, samples] contract."""
    class _StubEnhancer:
        def enhance(self, waveform, in_sr):  # pragma: no cover - not reached
            return waveform

    with pytest.raises(ValueError):
        _denoise_reference_waveform(_StubEnhancer(), mx.zeros((2, 8)), 16_000)


def _make_bare_model() -> DramaBoxModel:
    """Construct a DramaBoxModel without loading any heavy weights.

    Only the RE-USE lazy-load path is exercised, so the model components can be
    placeholders.
    """
    return DramaBoxModel(
        prompt_encoder=None,
        dit=None,
        audio_vae=None,
        vocoder=None,
    )


def test_missing_reuse_weights_raises_clear_error(monkeypatch):
    """`denoise_ref=True` with RE-USE unavailable raises a clear, named error.

    Simulate the resolver failing (no weights / no network). The error must name
    RE-USE and the `denoise_ref=False` opt-out, and must NOT silently skip.
    """

    def _boom(*_args, **_kwargs):
        raise FileNotFoundError("no RE-USE weights here")

    monkeypatch.setattr(dramabox_mod, "resolve_reuse_path", _boom)

    model = _make_bare_model()
    with pytest.raises(RuntimeError) as excinfo:
        model._get_reuse_enhancer()

    message = str(excinfo.value)
    assert "RE-USE" in message
    assert "denoise_ref=False" in message


def test_reuse_enhancer_is_cached(monkeypatch):
    """The lazily-loaded enhancer is built once and reused."""

    class _StubEnhancer:
        pass

    built = []

    def _fake_resolve(_arg):
        return "/tmp/does-not-matter"

    def _fake_from_dir(_path):
        inst = _StubEnhancer()
        built.append(inst)
        return inst

    monkeypatch.setattr(dramabox_mod, "resolve_reuse_path", _fake_resolve)
    monkeypatch.setattr(
        dramabox_mod.REUSEEnhancer, "from_dir", staticmethod(_fake_from_dir)
    )

    model = _make_bare_model()
    first = model._get_reuse_enhancer()
    second = model._get_reuse_enhancer()

    assert first is second
    assert len(built) == 1
