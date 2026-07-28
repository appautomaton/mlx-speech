"""Runtime-purity and constant-work checks for Nemotron ASR."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import mlx.core as mx
import pytest

import mlx_speech.asr as asr
from mlx_speech.models.nemotron_asr.model import NemotronASRModel
from mlx_speech.models.nemotron_asr.streaming import StreamingEncoder
from mlx_speech.models.nemotron_asr.subsampling import subsampled_length

CHECKPOINT = Path("models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16")
RUNTIME_PACKAGE = Path("src/mlx_speech/models/nemotron_asr")
FORBIDDEN = frozenset({"torch", "nemo", "transformers"})

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(),
    reason="converted Nemotron checkpoint not present",
)


def test_runtime_source_has_no_forbidden_imports() -> None:
    violations = []
    for path in sorted(RUNTIME_PACKAGE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots = [node.module.split(".")[0]]
            else:
                continue
            for root in roots:
                if root in FORBIDDEN:
                    violations.append(f"{path}:{node.lineno}: {root}")
    assert violations == []


def test_public_local_load_and_inference_import_no_forbidden_runtime() -> None:
    before = set(sys.modules)

    adapter = asr.load(str(CHECKPOINT))
    session = adapter.stream_session(language="en-US", att_context_size=(56, 3))
    session.feed(mx.zeros((4_001,), dtype=mx.float32))
    session.finalize()

    imported = set(sys.modules) - before
    forbidden = sorted(
        name for name in imported if name.split(".")[0] in FORBIDDEN
    )
    assert forbidden == []


@pytest.mark.parametrize("mel_frames", [256, 512, 1024])
def test_each_encoder_frame_visits_each_block_exactly_once(mel_frames: int) -> None:
    model = NemotronASRModel.from_dir(CHECKPOINT)
    stream = StreamingEncoder(model.encoder, att_context_size=(56, 13))
    features = mx.zeros((1, mel_frames, 128), dtype=mx.float32)

    output = stream.feed(features, final=True)
    output_frames = sum(chunk.shape[1] for chunk in output)

    assert output_frames == subsampled_length(mel_frames)
    assert stream.block_frame_evaluations == output_frames * 24
