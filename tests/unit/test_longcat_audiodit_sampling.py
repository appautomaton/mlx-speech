from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from mlx_speech.models.longcat_audiodit.config import LongCatAudioDiTConfig
from mlx_speech.models.longcat_audiodit.model import LongCatAudioDiTModel
from mlx_speech.models.longcat_audiodit.sampling import _MomentumBuffer, odeint_euler


def test_odeint_euler_integrates_linearly() -> None:
    y0 = mx.array([[0.0]], dtype=mx.float32)
    t = mx.array([0.0, 0.5, 1.0], dtype=mx.float32)

    trajectory = odeint_euler(
        lambda current_t, y: mx.ones_like(y) * (1.0 + current_t),
        y0,
        t,
    )

    assert trajectory.shape == (3, 1, 1)
    assert float(trajectory[-1, 0, 0]) > 1.0


def test_momentum_buffer_tracks_running_average() -> None:
    buffer = _MomentumBuffer(momentum=-0.3)
    buffer.update(mx.array([1.0], dtype=mx.float32))
    first = float(buffer.running_average[0])
    buffer.update(mx.array([2.0], dtype=mx.float32))
    second = float(buffer.running_average[0])
    assert second != first


@dataclass(frozen=True)
class _FakeEncoderOutput:
    last_hidden_state: mx.array
    hidden_states: tuple[mx.array, ...]


class _FakeTextEncoder:
    def __call__(
        self,
        *,
        input_ids: mx.array,
        attention_mask: mx.array,
        output_hidden_states: bool = False,
    ):
        del input_ids, attention_mask, output_hidden_states
        hidden = mx.ones((1, 3, 4), dtype=mx.float32)
        return _FakeEncoderOutput(last_hidden_state=hidden, hidden_states=(hidden,))


class _FakeTransformer:
    def __call__(
        self,
        *,
        x: mx.array,
        text: mx.array,
        text_len: mx.array,
        time: mx.array,
        mask: mx.array | None = None,
        cond_mask: mx.array | None = None,
        return_ith_layer: int | None = None,
        latent_cond: mx.array | None = None,
    ):
        del text, text_len, time, mask, cond_mask, return_ith_layer, latent_cond
        return {"last_hidden_state": x + 1.0, "hidden_state": None}


class _FakeVae:
    def __init__(self) -> None:
        self.decoded_inputs: list[mx.array] = []

    def encode(self, audio: mx.array) -> mx.array:
        frames = audio.shape[-1] // 4
        values = mx.arange(audio.shape[0] * 4 * frames, dtype=mx.float32)
        return values.reshape(audio.shape[0], 4, frames)

    def decode(self, latents: mx.array) -> mx.array:
        self.decoded_inputs.append(latents)
        return latents[:, :1, :]


def test_model_encode_prompt_audio_applies_off_three_trim() -> None:
    model = LongCatAudioDiTModel(
        LongCatAudioDiTConfig(latent_dim=4, latent_hop=4),
        text_encoder=_FakeTextEncoder(),
        transformer=_FakeTransformer(),
        vae=_FakeVae(),
    )

    prompt = mx.zeros((1, 1, 10), dtype=mx.float32)
    latent, prompt_dur = model.encode_prompt_audio(prompt)

    assert prompt_dur == 3
    assert latent.shape == (1, 3, 4)


def test_model_forward_trims_prompt_frames_before_decode() -> None:
    vae = _FakeVae()
    model = LongCatAudioDiTModel(
        LongCatAudioDiTConfig(latent_dim=4, latent_hop=4, max_wav_duration=2),
        text_encoder=_FakeTextEncoder(),
        transformer=_FakeTransformer(),
        vae=vae,
    )

    output = model(
        input_ids=mx.array([[1, 2, 3]], dtype=mx.int32),
        attention_mask=mx.array([[1, 1, 1]], dtype=mx.int32),
        prompt_audio=mx.zeros((1, 1, 10), dtype=mx.float32),
        duration=5,
        steps=2,
        cfg_strength=0.0,
        guidance_method="cfg",
    )

    assert output.latent.shape == (1, 4, 2)
    assert output.waveform.shape == (1, 2)
    assert vae.decoded_inputs[-1].shape == (1, 4, 2)
