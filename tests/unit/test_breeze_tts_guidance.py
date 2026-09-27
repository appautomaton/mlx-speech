"""Official single-branch CFG is combined before the repetition penalty."""

from __future__ import annotations

import mlx.core as mx

from mlx_speech.generation.breeze_tts import (
    _backbone_scores,
    _guide,
    _prompt_text,
)


def test_instruction_prompt_matches_the_official_span() -> None:
    assert _prompt_text("S0", "Hello.", None) == "[S0]Hello."
    assert _prompt_text("[S0]", "target", "Speak warmly.") == (
        "[S0]<ins_bos>Speak warmly.<ins_eos>target"
    )


def test_guidance_penalizes_the_combined_logits() -> None:
    conditional = mx.array([[-2.0, 3.0]])
    unconditional = mx.array([[4.0, -1.0]])
    guided = _backbone_scores(conditional, unconditional, 4.0, [0], 2.0)
    # token 0: 4 + 4 * (-2 - 4) = -20, then a negative logit is multiplied.
    assert guided[0, 0].item() == -40.0
    assert guided[0, 1].item() == 15.0


def test_unit_scale_keeps_the_conditional_branch() -> None:
    conditional = mx.array([[1.0, -2.0]], dtype=mx.bfloat16)
    unconditional = mx.array([[9.0, 9.0]], dtype=mx.bfloat16)
    guided = _guide(conditional, unconditional, 1.0)
    assert guided.dtype == mx.float32
    assert mx.allclose(guided, mx.array([[1.0, -2.0]]))


def test_reserved_codec_ids_stay_masked_after_guidance() -> None:
    conditional = mx.zeros((1, 2052))
    unconditional = mx.zeros((1, 2052))
    conditional[0, 2048] = 100.0
    unconditional[0, 2048] = 50.0
    conditional[0, 2051] = 5.0
    unconditional[0, 2051] = 1.0
    guided = _backbone_scores(conditional, unconditional, 4.0, [], 1.1)
    assert guided[0, 2048].item() == -1.0e9
    assert guided[0, 2051].item() == 1.0 + 4.0 * (5.0 - 1.0)
