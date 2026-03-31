from __future__ import annotations

import pytest

from mlx_voice.generation import MossTTSDelayGenerationConfig, generate_moss_tts_delay
from mlx_voice.models.moss_delay import MossTTSDelayProcessor, load_moss_tts_delay_model

pytestmark = pytest.mark.local_integration


@pytest.fixture(scope="module")
def real_delay_runtime():
    loaded = load_moss_tts_delay_model()
    processor = MossTTSDelayProcessor.from_path(loaded.model_dir)
    return loaded.model, processor


def _run_real_delay_greedy(
    model,
    processor: MossTTSDelayProcessor,
    text: str,
    *,
    max_new_tokens: int,
    use_kv_cache: bool,
):
    batch = processor(
        [[processor.build_user_message(text=text)]],
        mode="generation",
    )
    return generate_moss_tts_delay(
        model,
        batch.input_ids,
        batch.attention_mask,
        config=MossTTSDelayGenerationConfig(
            max_new_tokens=max_new_tokens,
            text_temperature=0.0,
            audio_temperature=0.0,
            do_sample=False,
            use_kv_cache=use_kv_cache,
        ),
    )


def _assert_real_generation_output(output, *, max_new_tokens: int) -> None:
    assert output.generated_rows.ndim == 3
    assert 0 < output.generated_rows.shape[1] <= max_new_tokens
    assert len(output.messages) == 1
    assert output.messages[0][1].ndim == 2
    assert output.messages[0][1].shape[0] > 0


def test_delay_real_runtime_generates_without_kv_cache(real_delay_runtime) -> None:
    model, processor = real_delay_runtime
    output = _run_real_delay_greedy(
        model,
        processor,
        "[S1] Watson, we should go now.",
        max_new_tokens=48,
        use_kv_cache=False,
    )

    _assert_real_generation_output(output, max_new_tokens=48)


def test_delay_real_runtime_generates_with_kv_cache(real_delay_runtime) -> None:
    model, processor = real_delay_runtime
    output = _run_real_delay_greedy(
        model,
        processor,
        (
            "[S1] Watson, should we go now? The rain is getting worse, and the road is "
            "flooding. Can you hear the thunder?"
        ),
        max_new_tokens=80,
        use_kv_cache=True,
    )

    _assert_real_generation_output(output, max_new_tokens=80)
