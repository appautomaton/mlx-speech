from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

from mlx_speech.generation.step_audio_editx import StepAudioEditXResult

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "generate_step_audio_editx.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("generate_step_audio_editx_script", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module from {SCRIPT_PATH}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_help_documents_clone_and_edit_modes() -> None:
    module = _load_script_module()

    help_text = module._build_parser().format_help()

    assert "clone" in help_text
    assert "edit" in help_text
    assert "--flow-steps" in help_text
    assert "--temperature" in help_text
    assert "--prefer-mlx-int8" in help_text


def test_parser_accepts_clone_arguments() -> None:
    module = _load_script_module()

    args = module._build_parser().parse_args(
        [
            "--prompt-audio",
            "ref.wav",
            "--prompt-text",
            "Prompt text.",
            "--output",
            "out.wav",
            "clone",
            "--target-text",
            "New text.",
        ]
    )

    assert args.mode == "clone"
    assert args.target_text == "New text."
    assert args.flow_steps == 10
    assert args.temperature == 0.7
    assert args.prefer_mlx_int8 is False


def test_parser_accepts_edit_arguments() -> None:
    module = _load_script_module()

    args = module._build_parser().parse_args(
        [
            "--prompt-audio",
            "ref.wav",
            "--prompt-text",
            "Prompt text.",
            "--output",
            "out.wav",
            "edit",
            "--edit-type",
            "style",
            "--edit-info",
            "whispering",
        ]
    )

    assert args.mode == "edit"
    assert args.edit_type == "style"
    assert args.edit_info == "whispering"


def test_parser_can_opt_into_mlx_int8_runtime() -> None:
    module = _load_script_module()

    args = module._build_parser().parse_args(
        [
            "--prompt-audio",
            "ref.wav",
            "--prompt-text",
            "Prompt text.",
            "--output",
            "out.wav",
            "--prefer-mlx-int8",
            "clone",
            "--target-text",
            "New text.",
        ]
    )

    assert args.prefer_mlx_int8 is True


def test_main_prints_elapsed_sec_and_rtf(monkeypatch, capsys) -> None:
    module = _load_script_module()

    result = StepAudioEditXResult(
        waveform=np.zeros((240,), dtype=np.float32),
        sample_rate=24000,
        generated_token_ids=[1, 2],
        generated_step1_token_ids=[11, 12, 3],
        generated_dual_timesteps=3,
        generated_mel_frames=6,
        expected_duration_seconds=0.12,
        stop_reached=True,
        stop_reason="eos",
        mode="clone",
        elapsed_sec=1.25,
        rtf=0.5,
    )

    fake_model = SimpleNamespace(clone=lambda *args, **kwargs: result)
    monkeypatch.setattr(module, "load_audio", lambda path, mono=True: (np.zeros((1600,), dtype=np.float32), 16000))
    monkeypatch.setattr(module.StepAudioEditXModel, "from_dir", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(module, "write_wav", lambda path, waveform, sample_rate: Path(path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--prompt-audio",
            "ref.wav",
            "--prompt-text",
            "Prompt text.",
            "--output",
            "out.wav",
            "clone",
            "--target-text",
            "New text.",
        ],
    )

    module.main()
    output = capsys.readouterr().out

    assert "elapsed_sec: 1.25" in output
    assert "RTF: 0.50x" in output
