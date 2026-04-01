from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

import mlx_speech.generation.step_audio_editx as generation_module
from mlx_speech.generation.step_audio_editx import (
    StepAudioEditXModel,
    StepAudioEditXResult,
)


def _logits_with_token(token_id: int, *, vocab_size: int = 32) -> mx.array:
    logits = np.full((1, 1, vocab_size), -1e9, dtype=np.float32)
    logits[0, 0, token_id] = 0.0
    return mx.array(logits, dtype=mx.float32)


class _FakeStep1Model:
    def __init__(self, generated_tokens: list[int], *, vocab_size: int = 32):
        self.generated_tokens = list(generated_tokens)
        self.vocab_size = vocab_size
        self.calls: list[list[int]] = []
        self._cursor = 0

    def allocate_kv_cache(self, *, batch_size: int, max_length: int, dtype=None):
        del batch_size, max_length, dtype
        return SimpleNamespace(current_length=0)

    def __call__(self, input_ids: mx.array, *, cache=None):
        if cache is not None:
            cache.current_length += int(np.asarray(input_ids).shape[-1])
        tokens = np.asarray(input_ids).reshape(-1).astype(np.int32).tolist()
        self.calls.append(tokens)
        next_token = self.generated_tokens[min(self._cursor, len(self.generated_tokens) - 1)]
        self._cursor += 1
        return SimpleNamespace(
            logits=_logits_with_token(next_token, vocab_size=self.vocab_size),
            cache=cache,
        )


class _FakeTextTokenizer:
    def __init__(self, *, audio_token_base_id: int = 10):
        self.audio_token_base_id = audio_token_base_id
        self.clone_calls: list[dict[str, str]] = []
        self.edit_calls: list[dict[str, str]] = []

    def token_to_id(self, token: str) -> int:
        if token != "<audio_0>":
            raise KeyError(token)
        return self.audio_token_base_id

    def build_clone_prompt_ids(
        self,
        *,
        speaker: str,
        prompt_text: str,
        prompt_wav_tokens: str,
        target_text: str,
    ) -> list[int]:
        self.clone_calls.append(
            {
                "speaker": speaker,
                "prompt_text": prompt_text,
                "prompt_wav_tokens": prompt_wav_tokens,
                "target_text": target_text,
            }
        )
        return [1, 2, 3]

    def build_edit_prompt_ids(
        self,
        *,
        instruct_prefix: str,
        audio_token_str: str,
    ) -> list[int]:
        self.edit_calls.append(
            {
                "instruct_prefix": instruct_prefix,
                "audio_token_str": audio_token_str,
            }
        )
        return [4, 5]


def _build_stub_wrapper(*, generated_tokens: list[int]) -> StepAudioEditXModel:
    step1 = SimpleNamespace(
        model=_FakeStep1Model(generated_tokens),
        config=SimpleNamespace(max_seq_len=64, eos_token_id=3),
    )
    tokenizer = _FakeTextTokenizer(audio_token_base_id=10)
    vq02 = SimpleNamespace(runtime=SimpleNamespace(encode=lambda audio, sample_rate: [1, 2]))
    vq06 = SimpleNamespace(runtime=SimpleNamespace(encode=lambda audio, sample_rate: [3, 4, 5]))
    frontend = SimpleNamespace(
        extract_speech_feat=lambda audio, sample_rate: (np.ones((1, 4, 80), dtype=np.float32), np.array([4], dtype=np.int64)),
        extract_spk_embedding=lambda audio, sample_rate: np.ones((1, 192), dtype=np.float32),
    )
    conditioner = SimpleNamespace(
        model=SimpleNamespace(
            prepare_nonstream_inputs=lambda **kwargs: SimpleNamespace(
                **kwargs,
                token_dual=np.zeros((1, 3, 2), dtype=np.int64),
            ),
        )
    )
    flow = SimpleNamespace(
        model=SimpleNamespace(
            inference=lambda prepared, n_timesteps: np.ones((1, 80, 6), dtype=np.float32),
        )
    )
    hift = SimpleNamespace(
        config=SimpleNamespace(
            sampling_rate=24000,
            upsample_rates=(8, 5, 3),
            istft_hop_len=4,
        ),
        model=SimpleNamespace(
            inference=lambda mel: (np.ones((1, 240), dtype=np.float32), np.zeros((1, 1, 240), dtype=np.float32)),
        ),
    )
    return StepAudioEditXModel(
        step1=step1,
        tokenizer=tokenizer,
        tokenizer_dir=Path("models/stepfun/step_audio_tokenizer/original"),
        vq02=vq02,
        vq06=vq06,
        frontend=frontend,
        conditioner=conditioner,
        flow=flow,
        hift=hift,
    )


def test_clone_returns_waveform_result_and_eos_stop() -> None:
    wrapper = _build_stub_wrapper(generated_tokens=[11, 12, 3])

    result = wrapper.clone(
        np.zeros(1600, dtype=np.float32),
        16000,
        "Prompt text.",
        "Target text.",
        temperature=0.0,
    )

    assert isinstance(result, StepAudioEditXResult)
    assert result.sample_rate == 24000
    assert result.generated_token_ids == [1, 2]
    assert result.generated_step1_token_ids == [11, 12, 3]
    assert result.generated_dual_timesteps == 3
    assert result.generated_mel_frames == 6
    assert result.expected_duration_seconds == pytest.approx(0.12)
    assert result.stop_reached is True
    assert result.stop_reason == "eos"
    assert result.mode == "clone"
    assert result.waveform.shape == (240,)
    assert wrapper.tokenizer.clone_calls[0]["speaker"] == "debug"
    assert "<audio_" in wrapper.tokenizer.clone_calls[0]["prompt_wav_tokens"]


def test_edit_builds_instruction_and_reports_max_token_stop() -> None:
    wrapper = _build_stub_wrapper(generated_tokens=[11, 12, 13])

    result = wrapper.edit(
        np.zeros(800, dtype=np.float32),
        16000,
        "Reference text.",
        "style",
        edit_info="whispering",
        max_new_tokens=2,
        temperature=0.0,
    )

    assert result.generated_token_ids == [1, 2]
    assert result.generated_step1_token_ids == [11, 12]
    assert result.generated_dual_timesteps == 3
    assert result.generated_mel_frames == 6
    assert result.expected_duration_seconds == pytest.approx(0.12)
    assert result.stop_reached is False
    assert result.stop_reason == "max_new_tokens"
    assert result.mode == "edit"
    assert "whispering style" in wrapper.tokenizer.edit_calls[0]["instruct_prefix"]
    assert "<audio_" in wrapper.tokenizer.edit_calls[0]["audio_token_str"]


def test_clone_skips_non_audio_tokens_and_keeps_collecting_audio_ids() -> None:
    wrapper = _build_stub_wrapper(generated_tokens=[5, 11, 12, 3])

    result = wrapper.clone(
        np.zeros(1600, dtype=np.float32),
        16000,
        "Prompt text.",
        "Target text.",
        temperature=0.0,
    )

    assert result.generated_token_ids == [1, 2]
    assert result.generated_step1_token_ids == [5, 11, 12, 3]
    assert result.generated_dual_timesteps == 3
    assert result.generated_mel_frames == 6
    assert result.expected_duration_seconds == pytest.approx(0.12)
    assert result.stop_reached is True
    assert result.stop_reason == "eos"
    assert len(wrapper.step1.model.calls) == 4


def test_clone_stops_on_step1_eot_not_tokenizer_eos() -> None:
    wrapper = _build_stub_wrapper(generated_tokens=[11, 3, 11])
    wrapper.tokenizer.eos_token_id = 2

    result = wrapper.clone(
        np.zeros(1600, dtype=np.float32),
        16000,
        "Prompt text.",
        "Target text.",
        temperature=0.0,
    )

    assert result.generated_token_ids == [1]
    assert result.generated_step1_token_ids == [11, 3]
    assert result.stop_reached is True
    assert result.stop_reason == "eos"


def test_clone_reports_elapsed_sec_and_rtf(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = _build_stub_wrapper(generated_tokens=[11, 12, 3])
    perf_counter_values = iter([10.0, 10.5])
    monkeypatch.setattr(generation_module.time, "perf_counter", lambda: next(perf_counter_values))

    result = wrapper.clone(
        np.zeros(1600, dtype=np.float32),
        16000,
        "Prompt text.",
        "Target text.",
        temperature=0.0,
    )

    assert result.elapsed_sec == pytest.approx(0.5)
    assert result.rtf == pytest.approx((240 / 24000) / 0.5)


def test_from_dir_loads_all_runtime_components(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    module = __import__(
        "mlx_speech.generation.step_audio_editx",
        fromlist=["StepAudioEditXModel"],
    )

    step1_loaded = SimpleNamespace(model_dir=Path("/tmp/step_audio_editx/original"), config=SimpleNamespace(max_seq_len=32768))
    tokenizer_loaded = SimpleNamespace()
    vq02_loaded = SimpleNamespace()
    vq06_loaded = SimpleNamespace()
    frontend_loaded = SimpleNamespace()
    conditioner_loaded = SimpleNamespace()
    flow_loaded = SimpleNamespace()
    hift_loaded = SimpleNamespace()

    def fake_load_step1(model_dir, prefer_mlx_int8, strict):
        calls["step1"] = (model_dir, prefer_mlx_int8, strict)
        return step1_loaded

    def fake_resolve_tokenizer_dir(tokenizer_dir):
        del tokenizer_dir
        calls["tokenizer_dir"] = Path("/tmp/step_audio_tokenizer/original")
        return calls["tokenizer_dir"]

    def fake_text_tokenizer_from_path(cls, path):
        del cls
        calls["text_tokenizer"] = Path(path)
        return tokenizer_loaded

    def fake_load_vq02(path, strict):
        calls["vq02"] = (Path(path), strict)
        return vq02_loaded

    def fake_load_vq06(path, strict):
        calls["vq06"] = (Path(path), strict)
        return vq06_loaded

    def fake_frontend_from_model_dir(cls, path):
        del cls
        calls["frontend"] = Path(path)
        return frontend_loaded

    def fake_conditioner(path):
        calls["conditioner"] = Path(path)
        return conditioner_loaded

    def fake_flow(path):
        calls["flow"] = Path(path)
        return flow_loaded

    def fake_hift(path):
        calls["hift"] = Path(path)
        return hift_loaded

    monkeypatch.setattr(module, "load_step_audio_editx_model", fake_load_step1)
    monkeypatch.setattr(module, "resolve_step_audio_tokenizer_model_dir", fake_resolve_tokenizer_dir)
    monkeypatch.setattr(module.StepAudioEditXTokenizer, "from_path", classmethod(fake_text_tokenizer_from_path))
    monkeypatch.setattr(module, "load_step_audio_vq02_model", fake_load_vq02)
    monkeypatch.setattr(module, "load_step_audio_vq06_model", fake_load_vq06)
    monkeypatch.setattr(module.StepAudioCosyVoiceFrontEnd, "from_model_dir", classmethod(fake_frontend_from_model_dir))
    monkeypatch.setattr(module, "load_step_audio_flow_conditioner", fake_conditioner)
    monkeypatch.setattr(module, "load_step_audio_flow_model", fake_flow)
    monkeypatch.setattr(module, "load_step_audio_hift_model", fake_hift)

    loaded = StepAudioEditXModel.from_dir(None, strict=False)

    assert loaded.step1 is step1_loaded
    assert loaded.tokenizer is tokenizer_loaded
    assert loaded.vq02 is vq02_loaded
    assert loaded.vq06 is vq06_loaded
    assert loaded.frontend is frontend_loaded
    assert loaded.conditioner is conditioner_loaded
    assert loaded.flow is flow_loaded
    assert loaded.hift is hift_loaded
    assert calls["step1"] == (None, False, False)
    assert calls["text_tokenizer"] == step1_loaded.model_dir
    assert calls["frontend"] == step1_loaded.model_dir
