from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.audio import write_wav
from mlx_speech.generation import (
    MossTTSLocalGenerationConfig,
    synthesize_moss_tts_local_conversations,
)
from mlx_speech.models.moss_audio_tokenizer import load_moss_audio_tokenizer_model
from mlx_speech.models.moss_local import MossTTSLocalProcessor, load_moss_tts_local_model

pytestmark = pytest.mark.local_integration


def _runtime():
    loaded_model = load_moss_tts_local_model()
    loaded_codec = load_moss_audio_tokenizer_model()
    processor = MossTTSLocalProcessor.from_path(
        loaded_model.model_dir,
        audio_tokenizer=loaded_codec.model,
    )
    return loaded_model.model, processor, loaded_codec.model


@pytest.fixture()
def reference_audio_path(tmp_path: Path) -> Path:
    sample_rate = 24000
    t = mx.arange(0, 4800, dtype=mx.float32) / float(sample_rate)
    waveform = 0.1 * mx.sin(t * 2.0 * 3.14159265 * 220.0)
    path = tmp_path / "reference.wav"
    write_wav(path, waveform, sample_rate=sample_rate)
    return path


@pytest.mark.parametrize(
    ("mode", "processor_mode"),
    [
        ("generation", "generation"),
        ("clone", "generation"),
        ("continuation", "continuation"),
        ("continue_clone", "continuation"),
    ],
)
def test_default_quantized_runtime_supports_main_inference_modes(
    reference_audio_path: Path,
    mode: str,
    processor_mode: str,
) -> None:
    model, processor, codec = _runtime()
    config = MossTTSLocalGenerationConfig(max_new_tokens=2, do_sample=False)

    user_kwargs: dict[str, object] = {"text": "Hello from MLX.", "tokens": 8}
    if mode in {"clone", "continue_clone"}:
        user_kwargs["reference"] = [str(reference_audio_path)]

    if mode == "generation":
        conversations = [[processor.build_user_message(**user_kwargs)]]
    elif mode == "clone":
        conversations = [[processor.build_user_message(**user_kwargs)]]
    elif mode == "continuation":
        conversations = [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[str(reference_audio_path)]),
            ]
        ]
    else:
        conversations = [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[str(reference_audio_path)]),
            ]
        ]

    result = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=conversations,
        mode=processor_mode,
        config=config,
    )

    assert len(result.outputs) == 1
    assert result.outputs[0].waveform.ndim == 1
    assert result.outputs[0].waveform.size > 0
    assert result.outputs[0].sample_rate == 24000


def test_batch_inference_preserves_output_order(reference_audio_path: Path) -> None:
    model, processor, codec = _runtime()
    config = MossTTSLocalGenerationConfig(max_new_tokens=2, do_sample=False)
    conversations = [
        [processor.build_user_message(text="First sample.", tokens=6)],
        [processor.build_user_message(text="Second sample.", reference=[str(reference_audio_path)], tokens=6)],
    ]

    result = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=conversations,
        mode="generation",
        config=config,
    )

    assert len(result.outputs) == 2
    assert result.outputs[0].waveform.size > 0
    assert result.outputs[1].waveform.size > 0


@pytest.mark.parametrize(
    ("mode", "processor_mode"),
    [
        ("generation", "generation"),
        ("clone", "generation"),
        ("continuation", "continuation"),
        ("continue_clone", "continuation"),
    ],
)
def test_cached_and_uncached_single_sample_paths_match(
    reference_audio_path: Path,
    mode: str,
    processor_mode: str,
) -> None:
    model, processor, codec = _runtime()

    user_kwargs: dict[str, object] = {"text": "Cache parity sample.", "tokens": 8}
    if mode in {"clone", "continue_clone"}:
        user_kwargs["reference"] = [str(reference_audio_path)]

    if mode == "generation":
        conversations = [[processor.build_user_message(**user_kwargs)]]
    elif mode == "clone":
        conversations = [[processor.build_user_message(**user_kwargs)]]
    elif mode == "continuation":
        conversations = [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[str(reference_audio_path)]),
            ]
        ]
    else:
        conversations = [
            [
                processor.build_user_message(**user_kwargs),
                processor.build_assistant_message(audio_codes_list=[str(reference_audio_path)]),
            ]
        ]

    cached = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=conversations,
        mode=processor_mode,
        config=MossTTSLocalGenerationConfig(max_new_tokens=2, do_sample=False, use_kv_cache=True),
    )
    uncached = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=conversations,
        mode=processor_mode,
        config=MossTTSLocalGenerationConfig(
            max_new_tokens=2,
            do_sample=False,
            use_kv_cache=False,
        ),
    )

    assert cached.generation.sequences.tolist() == uncached.generation.sequences.tolist()
    assert cached.generation.generated_rows.tolist() == uncached.generation.generated_rows.tolist()
    assert cached.generation.audio_codes_list[0].tolist() == uncached.generation.audio_codes_list[0].tolist()


def test_batch_kv_cache_flag_falls_back_to_uncached_path(
    reference_audio_path: Path,
) -> None:
    model, processor, codec = _runtime()
    conversations = [
        [processor.build_user_message(text="First cached batch item.", tokens=6)],
        [processor.build_user_message(text="Second cached batch item.", reference=[str(reference_audio_path)], tokens=6)],
    ]

    cached = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=conversations,
        mode="generation",
        config=MossTTSLocalGenerationConfig(max_new_tokens=2, do_sample=False, use_kv_cache=True),
    )
    uncached = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=conversations,
        mode="generation",
        config=MossTTSLocalGenerationConfig(
            max_new_tokens=2,
            do_sample=False,
            use_kv_cache=False,
        ),
    )

    assert cached.generation.sequences.tolist() == uncached.generation.sequences.tolist()
    assert len(cached.outputs) == 2
    assert cached.outputs[0].waveform.size > 0
    assert cached.outputs[1].waveform.size > 0


def test_default_cached_path_runs() -> None:
    model, processor, codec = _runtime()
    conversations = [[processor.build_user_message(text="Local cache sample.", tokens=6)]]

    mx.random.seed(0)
    result = synthesize_moss_tts_local_conversations(
        model,
        processor,
        codec,
        conversations=conversations,
        mode="generation",
        config=MossTTSLocalGenerationConfig(
            max_new_tokens=2,
            use_kv_cache=True,
        ),
    )

    assert len(result.outputs) == 1
    assert result.outputs[0].waveform.ndim == 1
    assert result.outputs[0].waveform.size > 0
