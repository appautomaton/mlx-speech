"""End-to-end greedy decode parity for the Nemotron bf16 checkpoint."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_speech.audio import load_audio
from mlx_speech.models.nemotron_asr.checkpoint import (
    load_nemotron_checkpoint,
    load_state_dict_strict,
)
from mlx_speech.models.nemotron_asr.model import NemotronASRModel
from mlx_speech.models.nemotron_asr.prompt import apply_language_prompt

CHECKPOINT = Path("models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16")
# FLEURS en-US speech fixture from pinned mlx-audio d28d68c6. Reference text:
# "Then Lakasang took the lead in singing the passions."
CLIP = Path(".references/mlx-audio/mlx_audio/stt/tests/mega_asr/fixtures/clean.wav")
REFERENCE_SOURCE = Path(
    ".references/mlx-audio/mlx_audio/stt/models/nemotron_asr"
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file() or not CLIP.is_file(),
    reason="Nemotron checkpoint or pinned speech fixture not present",
)

EXPECTED_TOKENS = (
    2845,
    113,
    2,
    214,
    46,
    329,
    2,
    193,
    47,
    2959,
    85,
    12945,
    46,
    1305,
    26,
    274,
    2812,
    2,
    819,
    2959,
    47,
    2959,
    150,
    538,
    1388,
    85,
    131,
    40,
    4,
    2,
    2947,
)
EXPECTED_TEXT = "Then Loka Singh to delete and singing the Bashans."
EXPECTED_AUTO_TOKENS = (
    2845,
    113,
    2,
    214,
    46,
    329,
    2,
    193,
    47,
    2959,
    85,
    12945,
    46,
    1305,
    26,
    274,
    2812,
    2,
    819,
    2959,
    47,
    2959,
    150,
    9599,
    40,
    85,
    131,
    40,
    4,
    2,
    2947,
)
EXPECTED_AUTO_TEXT = "Then Loka Singh to delete and singing the Pashans."


@pytest.fixture(scope="module")
def runtime() -> tuple[NemotronASRModel, mx.array]:
    waveform, sample_rate = load_audio(CLIP, sample_rate=16_000, mono=True)
    assert sample_rate == 16_000
    return NemotronASRModel.from_dir(CHECKPOINT), waveform


def _load_reference_modules() -> tuple[types.ModuleType, types.ModuleType]:
    package_name = "_mlx_audio_nemotron_runtime_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REFERENCE_SOURCE.resolve())]  # type: ignore[attr-defined]
    sys.modules[package_name] = package
    modules = {}
    for name in ("config", "attention", "conformer", "rnnt"):
        full_name = f"{package_name}.{name}"
        spec = importlib.util.spec_from_file_location(
            full_name, REFERENCE_SOURCE / f"{name}.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load mlx-audio reference module {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)
        modules[name] = module
    return modules["conformer"], modules["rnnt"]


def _reference_tokens(features: mx.array, lengths: mx.array) -> tuple[int, ...]:
    checkpoint = load_nemotron_checkpoint(CHECKPOINT)
    conformer_module, rnnt_module = _load_reference_modules()

    class ReferenceModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            args = checkpoint.config
            self.encoder = conformer_module.Conformer(args.encoder)
            self.prompt_kernel = [
                nn.Linear(
                    args.encoder.d_model + args.prompt.num_prompts,
                    args.prompt.prompt_hidden,
                ),
                nn.ReLU(),
                nn.Linear(args.prompt.prompt_hidden, args.encoder.d_model),
            ]
            self.decoder = rnnt_module.PredictNetwork(args.decoder)
            self.joint = rnnt_module.JointNetwork(args.joint)

    reference = ReferenceModel()
    state = {
        key: value
        for key, value in checkpoint.state_dict.items()
        if not key.startswith("preprocessor.")
    }
    assert load_state_dict_strict(reference, state).is_exact_match

    dtype = reference.encoder.pre_encode.out.weight.dtype
    encoded, encoded_lengths = reference.encoder(
        features.astype(dtype), lengths, att_context_size=[56, 13]
    )
    encoded = apply_language_prompt(
        encoded,
        "en-US",
        checkpoint.config.prompt,
        reference.prompt_kernel,
    )
    mx.eval(encoded, encoded_lengths)

    blank = checkpoint.config.decoder.vocab_size
    last_token = blank
    decoder_state = None
    tokens = []
    time = 0
    symbols_at_frame = 0
    while time < int(encoded_lengths[0].item()):
        current = (
            None
            if last_token == blank
            else mx.array([[last_token]], dtype=mx.int32)
        )
        prediction, proposed_state = reference.decoder(current, decoder_state)
        prediction = prediction.astype(encoded.dtype)
        logits = reference.joint(encoded[:, time : time + 1], prediction)
        token = int(mx.argmax(logits).item())
        if token == blank:
            time += 1
            symbols_at_frame = 0
            continue
        tokens.append(token)
        last_token = token
        decoder_state = proposed_state
        symbols_at_frame += 1
        if symbols_at_frame >= checkpoint.config.max_symbols:
            time += 1
            symbols_at_frame = 0
    return tuple(tokens)


def test_language_specified_decode_matches_captured_reference(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, waveform = runtime

    result = model.transcribe(
        waveform,
        language="en-US",
        att_context_size=(56, 13),
    )

    assert result.tokens == EXPECTED_TOKENS
    assert result.text == EXPECTED_TEXT
    assert result.language == "en-US"
    assert result.detected_language == "en-US"


def test_auto_language_decode_emits_detected_tag(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, waveform = runtime

    result = model.transcribe(
        waveform,
        language="auto",
        att_context_size=(56, 13),
        strip_language_tags=False,
    )

    assert result.tokens == EXPECTED_AUTO_TOKENS
    assert result.text == f"{EXPECTED_AUTO_TEXT} <en-US>"
    assert result.language == "auto"
    assert result.detected_language == "en-US"


def test_tokens_match_live_pinned_mlx_audio_reference(
    runtime: tuple[NemotronASRModel, mx.array],
) -> None:
    model, waveform = runtime
    features, lengths = model.preprocessor(waveform)

    assert _reference_tokens(features, lengths) == EXPECTED_TOKENS
