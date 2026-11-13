"""Opt-in batch transcription through the public Qwen3-ASR adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from mlx_speech import asr
from mlx_speech.audio import load_audio


ROOT = Path(__file__).parents[2]
ARTIFACT = ROOT / "models/qwen3_asr_1_7b/mlx-int8"
ENGLISH_AUDIO = ROOT / "examples/audio/vibevoice_4speaker_convo.wav"
CHINESE_AUDIO = ROOT / "models/netease/confucius4_r2t2/samples/test.wav"
pytestmark = pytest.mark.integration


def test_qwen3_asr_public_api_transcribes_batch_in_input_order() -> None:
    if os.environ.get("RUN_LOCAL_INTEGRATION") != "1":
        pytest.skip("set RUN_LOCAL_INTEGRATION=1 for local checkpoint transcription")
    if not (ARTIFACT / "config.json").is_file() or not list(ARTIFACT.glob("*.safetensors")):
        pytest.skip("local Qwen3-ASR int8 artifact is unavailable")
    for path in (ENGLISH_AUDIO, CHINESE_AUDIO):
        if not path.is_file():
            pytest.skip(f"speech audio is unavailable: {path}")

    english, sr = load_audio(ENGLISH_AUDIO, sample_rate=16000, mono=True)
    chinese, chinese_sr = load_audio(CHINESE_AUDIO, sample_rate=16000, mono=True)
    assert sr == chinese_sr == 16000

    def crop(start: float, duration: float):
        return english[int(start * sr):int((start + duration) * sr)]

    # Deliberately unsorted lengths, mixed forced/automatic language prompts,
    # padding within groups, and more requests than the group size.
    clips = [crop(3.5, 11), chinese, crop(0, 2), crop(18, 7), crop(10, 4.5)]
    languages = ["English", "Chinese", None, "auto", ""]
    model = asr.load(str(ARTIFACT))
    results = model.generate_batch(clips, language=languages, batch_size=2)

    assert len(results) == len(clips)
    assert [result.language for result in results] == [
        "English", "Chinese", "English", "English", "English",
    ]
    assert all(result.text.strip() for result in results)
    # Content anchors identify the input clips without asserting equality with
    # serial decoding, exact punctuation, or a complete reference transcript.
    for index, anchor in ((0, "ramen"), (2, "lunch"), (3, "salad"), (4, "truck")):
        assert anchor in results[index].text.lower(), (index, results[index])
    assert any(0x4E00 <= ord(char) <= 0x9FFF for char in results[1].text)
