from __future__ import annotations

import mlx.core as mx

from mlx_speech.audio import write_wav
from mlx_speech.models.moss_delay.dialogue import (
    build_prefixed_ttsd_text,
    build_ttsd_conversation,
    collect_speaker_fields,
    normalize_ttsd_text,
    prepare_ttsd_sample,
    resolve_ttsd_processor_mode,
)


class _FakeProcessorConfig:
    sampling_rate = 24000
    n_vq = 16


class _FakeProcessor:
    def __init__(self) -> None:
        self.model_config = _FakeProcessorConfig()
        self.wav_calls: list[tuple[int, int]] = []

    def build_user_message(self, *, text: str, reference=None):
        return {"role": "user", "content": text, "reference": reference}

    def build_assistant_message(self, *, audio_codes_list):
        return {"role": "assistant", "content": "", "audio_codes_list": audio_codes_list}

    def encode_audios_from_wav(self, wav_list, sampling_rate: int, n_vq: int | None = None):
        self.wav_calls.append((len(wav_list), sampling_rate))
        return [
            mx.full((speaker_idx + 1, n_vq or self.model_config.n_vq), speaker_idx + 1, dtype=mx.int32)
            for speaker_idx in range(len(wav_list))
        ]


def test_normalize_ttsd_text_normalizes_tags_and_punctuation() -> None:
    text = "[1] Hello... [1] ha ha ha!!! [2] ——Wait??"

    normalized = normalize_ttsd_text(text)

    assert "[S1]" in normalized
    assert "[laugh]" in normalized
    assert "..." not in normalized


def test_collect_speaker_fields_resolves_only_paired_speakers() -> None:
    sample = {
        "prompt_audio_speaker1": "s1.wav",
        "prompt_text_speaker1": "[S1] Hello.",
        "prompt_audio_speaker2": "s2.wav",
        "prompt_text_speaker3": "[S3] Missing pair.",
    }

    audio_map, text_map, speaker_ids = collect_speaker_fields(sample)

    assert speaker_ids == [1]
    assert audio_map[1].endswith("s1.wav")
    assert text_map[1] == "[S1] Hello."


def test_build_prefixed_ttsd_text_merges_prompt_text_and_target_text() -> None:
    result = build_prefixed_ttsd_text(
        "[S1] Watson, we should leave now.",
        text_map={1: "The game is on, Watson."},
        speaker_ids=[1],
    )

    assert result.startswith("[S1]The game is on, Watson.")
    assert "Watson, we should leave now." in result


def test_resolve_ttsd_processor_mode_matches_upstream_shape() -> None:
    assert resolve_ttsd_processor_mode("generation") == "generation"
    assert resolve_ttsd_processor_mode("voice_clone") == "generation"
    assert resolve_ttsd_processor_mode("continuation") == "continuation"
    assert resolve_ttsd_processor_mode("voice_clone_and_continuation") == "continuation"


def test_build_ttsd_conversation_voice_clone_keeps_reference_list_shape(tmp_path) -> None:
    processor = _FakeProcessor()
    audio_path = tmp_path / "s1.wav"
    write_wav(audio_path, mx.zeros((240,), dtype=mx.float32), sample_rate=24000)
    conversation = build_ttsd_conversation(
        processor=processor,
        mode="voice_clone",
        text="[S1] Watson, we should leave now.",
        audio_map={1: str(audio_path)},
        text_map={1: "[S1] Prompt one."},
        n_vq=16,
    )

    assert len(conversation) == 1
    assert conversation[0]["role"] == "user"
    assert isinstance(conversation[0]["reference"], list)
    assert len(conversation[0]["reference"]) == 1


def test_build_ttsd_conversation_voice_clone_allows_audio_only_prompt_fields(tmp_path) -> None:
    processor = _FakeProcessor()
    audio_path = tmp_path / "s1.wav"
    write_wav(audio_path, mx.zeros((240,), dtype=mx.float32), sample_rate=24000)

    conversation = build_ttsd_conversation(
        processor=processor,
        mode="voice_clone",
        text="[S1] Watson, we should leave now.",
        audio_map={1: str(audio_path)},
        text_map={},
        n_vq=16,
    )

    assert len(conversation) == 1
    assert conversation[0]["role"] == "user"
    assert isinstance(conversation[0]["reference"], list)
    assert len(conversation[0]["reference"]) == 1


def test_prepare_ttsd_sample_builds_prefixed_continuation_conversation(tmp_path) -> None:
    processor = _FakeProcessor()
    audio_path = tmp_path / "s1.wav"
    write_wav(audio_path, mx.zeros((240,), dtype=mx.float32), sample_rate=24000)
    sample_id, output_record, conversation = prepare_ttsd_sample(
        line_no=7,
        raw_sample={
            "text": "[S1] Watson, we should leave now.",
            "prompt_audio_speaker1": str(audio_path),
            "prompt_text_speaker1": "[S1] The game is on, Watson.",
        },
        mode="continuation",
        processor=processor,
        n_vq=16,
    )

    assert sample_id == "000007"
    assert output_record["id"] == "000007"
    assert len(conversation) == 2
    assert "The game is on, Watson." in conversation[0]["content"]
    assert conversation[1]["role"] == "assistant"
