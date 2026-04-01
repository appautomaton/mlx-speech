"""Manual local integration smoke test for Step-Audio non-stream waveform inference."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
import soundfile as sf

from mlx_speech.models.step_audio_editx import (
    StepAudioCosyVoiceFrontEnd,
    StepAudioEditXTokenizer,
    load_step_audio_editx_model,
    load_step_audio_flow_conditioner,
    load_step_audio_flow_model,
    load_step_audio_hift_model,
)
from mlx_speech.models.step_audio_tokenizer import (
    format_audio_token_string,
    load_step_audio_vq02_model,
    load_step_audio_vq06_model,
    pack_raw_codes_to_prompt_tokens,
)
from tests.helpers.step_audio import EDITX_DIR, PROMPT_AUDIO, TOKENIZER_DIR, skip_no_integration

pytestmark = pytest.mark.integration


@skip_no_integration
def test_step_audio_nonstream_clone_path_produces_waveform() -> None:
    audio, sample_rate = sf.read(PROMPT_AUDIO, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    vq02 = load_step_audio_vq02_model(TOKENIZER_DIR)
    vq06 = load_step_audio_vq06_model(TOKENIZER_DIR)
    prompt_vq02 = vq02.runtime.encode(audio, sample_rate)
    prompt_vq06 = vq06.runtime.encode(audio, sample_rate)
    prompt_wav_tokens = format_audio_token_string(prompt_vq02, prompt_vq06)
    prompt_token = pack_raw_codes_to_prompt_tokens(prompt_vq02, prompt_vq06)

    tokenizer = StepAudioEditXTokenizer.from_path(EDITX_DIR)
    step1 = load_step_audio_editx_model(EDITX_DIR, prefer_mlx_int8=False)
    frontend = StepAudioCosyVoiceFrontEnd.from_model_dir(EDITX_DIR)
    conditioner = load_step_audio_flow_conditioner(EDITX_DIR)
    flow = load_step_audio_flow_model(EDITX_DIR)
    hift = load_step_audio_hift_model(EDITX_DIR)

    prompt_ids = tokenizer.build_clone_prompt_ids(
        speaker="debug",
        prompt_text="Loud is not allowed. Now you listen.",
        prompt_wav_tokens=prompt_wav_tokens,
        target_text="Testing the local nonstream waveform pipeline.",
    )

    outputs = step1.model(mx.array([prompt_ids], dtype=mx.int32))
    cache = outputs.cache
    next_logits = outputs.logits[:, -1, :]
    generated_audio_ids: list[int] = []
    for _ in range(128):
        next_token = int(mx.argmax(next_logits, axis=-1).item())
        if next_token >= 65536:
            generated_audio_ids.append(next_token - 65536)
            if len(generated_audio_ids) >= 6:
                break
        if next_token == step1.config.eos_token_id:
            break
        outputs = step1.model(mx.array([[next_token]], dtype=mx.int32), cache=cache)
        cache = outputs.cache
        next_logits = outputs.logits[:, -1, :]

    assert len(generated_audio_ids) >= 5

    prompt_feat, _ = frontend.extract_speech_feat(audio, sample_rate)
    speaker_embedding = frontend.extract_spk_embedding(audio, sample_rate)
    prepared = conditioner.model.prepare_nonstream_inputs(
        token=generated_audio_ids,
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        speaker_embedding=speaker_embedding,
    )
    mel = flow.model.inference(prepared, n_timesteps=2)
    waveform, _ = hift.model.inference(mel)

    assert hift.config.sampling_rate == 24000
    assert mel.shape[0] == 1
    assert mel.shape[1] == 80
    assert waveform.shape[0] == 1
    assert waveform.shape[1] > 0
    assert waveform.dtype == np.float32
    assert np.isfinite(waveform).all()
    assert float(np.abs(waveform).sum()) > 0.0
