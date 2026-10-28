from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel
from tokenizers.models import BPE

from mlx_speech.models.audio8_asr_infinite.checkpoint import (
    Audio8Checkpoint,
    load_checkpoint_into_model,
    sanitize_state_dict,
)
from mlx_speech.models.audio8_asr_infinite.config import Audio8Config
from mlx_speech.models.audio8_asr_infinite.model import Audio8ASRModel
from mlx_speech.models.audio8_asr_infinite.tokenizer import Audio8Tokenizer
from tests.unit.test_audio8_checkpoint import _config_payload, _source_weights


def _tokenizer() -> Audio8Tokenizer:
    special = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>",
        "[STREAMING_PAD]", "[STREAMING_WORD]", "[LANGUAGE_ZH]", "[LANGUAGE_EN]",
    ]
    vocabulary = {token: index for index, token in enumerate(special)}
    vocabulary.update({f"t{index}": index for index in range(len(special), 32)})
    tokenizer = Tokenizer(BPE(vocabulary, merges=[]))
    tokenizer.decoder = ByteLevel()
    tokenizer.add_special_tokens(special)
    return Audio8Tokenizer(
        tokenizer=tokenizer,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        streaming_pad_token_id=3,
        streaming_word_token_id=4,
        language_token_ids={"zh": 5, "en": 6},
    )


def _model() -> Audio8ASRModel:
    mx.random.seed(19)
    model = Audio8ASRModel(Audio8Config.from_dict(_config_payload()))
    model.tokenizer = _tokenizer()
    model.eval()
    return model


def test_assembled_model_strictly_accepts_complete_checkpoint():
    config = Audio8Config.from_dict(_config_payload())
    model = _model()
    state, _ = sanitize_state_dict(_source_weights(config), config)
    checkpoint = Audio8Checkpoint(Path("."), config, state, (), ())

    report = load_checkpoint_into_model(model, checkpoint, strict=True)

    assert report.checkpoint_only == ()
    assert report.shape_mismatches == ()
    assert set(report.model_only) == {
        "audio_tower.rotary_emb.inv_freq",
        "time_embedding.inv_freq",
    }
    assert model.language_model.lm_head is None


def test_arbitrary_pcm_packets_produce_identical_tokens_and_idempotent_finalize():
    model = _model()
    waveform = np.sin(np.arange(3200, dtype=np.float32) * 0.017)

    whole = model.stream_session(language="zh", frame_len=2, target_delay_ms=80)
    whole.feed(waveform)
    whole.finalize()
    whole_result = whole.result()

    packeted = model.stream_session(language="zh", frame_len=2, target_delay_ms=80)
    offset = 0
    for index, size in enumerate((1, 137, 400, 53, 160, 800)):
        packeted.feed(waveform[offset : offset + size])
        offset += size
    packeted.feed(waveform[offset:])
    packeted.finalize()

    assert packeted.result().generated_token_ids == whole_result.generated_token_ids
    assert packeted.result().text == whole_result.text
    assert packeted.finalize() == ()
    assert len(packeted.semantic_vad_probabilities) == len(whole_result.generated_token_ids)
    assert packeted.semantic_vad_probabilities[0].shape == (2, 8)
