"""Tests for VibeVoice generation loop."""

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_voice.models.vibevoice.config import VibeVoiceConfig
from mlx_voice.generation.vibevoice import (
    VibeVoiceGenerationConfig,
    VibeVoiceGenerationOutput,
    VibeVoiceSynthesisOutput,
    _constrain_logits,
)

MODEL_DIR = Path("models/vibevoice/mlx-int8")
HAS_INT8 = any(MODEL_DIR.glob("*.safetensors")) if MODEL_DIR.exists() else False
ORIGINAL_DIR = Path("models/vibevoice/original")
HAS_ORIGINAL = any(ORIGINAL_DIR.glob("*.safetensors")) if ORIGINAL_DIR.exists() else False
HAS_MODEL = HAS_INT8 or HAS_ORIGINAL


class TestConstrainLogits:
    def test_masks_invalid_tokens(self):
        logits = mx.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        valid = [1, 3]
        result = _constrain_logits(logits, valid)
        mx.eval(result)
        # Only positions 1 and 3 should have finite values
        assert float(result[0, 0]) == float("-inf")
        assert float(result[0, 1]) == 2.0
        assert float(result[0, 2]) == float("-inf")
        assert float(result[0, 3]) == 4.0
        assert float(result[0, 4]) == float("-inf")

    def test_argmax_selects_valid(self):
        logits = mx.array([[10.0, 1.0, 20.0, 5.0]])
        valid = [1, 3]
        result = _constrain_logits(logits, valid)
        best = mx.argmax(result, axis=-1).item()
        assert best == 3  # highest among valid tokens


class TestGenerationConfig:
    def test_defaults(self):
        cfg = VibeVoiceGenerationConfig()
        assert cfg.max_new_tokens == 4096
        assert cfg.cfg_scale == 1.3
        assert cfg.diffusion_steps == 20
        assert cfg.do_sample is False


@pytest.mark.skipif(not HAS_MODEL, reason="model not available")
class TestEndToEnd:
    def _get_model_dir(self):
        return MODEL_DIR if HAS_INT8 else ORIGINAL_DIR

    def test_short_generation(self):
        from mlx_voice.models.vibevoice.checkpoint import load_vibevoice_model
        from mlx_voice.models.vibevoice.tokenizer import VibeVoiceTokenizer
        from mlx_voice.generation.vibevoice import synthesize_vibevoice

        model_dir = self._get_model_dir()
        loaded = load_vibevoice_model(model_dir, strict=False)
        tok = VibeVoiceTokenizer.from_path(model_dir)
        config = VibeVoiceGenerationConfig(max_new_tokens=20, do_sample=False)

        result = synthesize_vibevoice(loaded.model, tok, "Hello.", config=config)
        mx.eval(result.waveform)

        assert isinstance(result, VibeVoiceSynthesisOutput)
        assert result.sample_rate == 24000
        assert result.generated_tokens > 0
        assert result.waveform.shape[0] > 0

    def test_voice_cloning(self):
        from mlx_voice.models.vibevoice.checkpoint import load_vibevoice_model
        from mlx_voice.models.vibevoice.tokenizer import VibeVoiceTokenizer
        from mlx_voice.generation.vibevoice import synthesize_vibevoice
        from mlx_voice.audio.io import load_audio

        ref_path = Path("outputs/source/hank_hill_ref.wav")
        if not ref_path.exists():
            pytest.skip("reference audio not available")

        model_dir = self._get_model_dir()
        loaded = load_vibevoice_model(model_dir, strict=False)
        tok = VibeVoiceTokenizer.from_path(model_dir)
        config = VibeVoiceGenerationConfig(max_new_tokens=20, do_sample=False)

        ref_raw, _ = load_audio(str(ref_path), sample_rate=24000)
        result = synthesize_vibevoice(
            loaded.model, tok, "Hello.",
            reference_audio=ref_raw.reshape(1, 1, -1),
            config=config,
        )
        mx.eval(result.waveform)
        assert result.waveform.shape[0] > 0
