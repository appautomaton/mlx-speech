"""Checkpoint loading test for `VocoderWithBWE`.

Verifies the 1224 vocoder + 3 mel_stft keys (= 1227 total under
``vocoder.*``) load with the expected shape remaps.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

from mlx_speech.models.dramabox.vocoder import (
    VocoderWithBWE,
    load_vocoder_with_bwe_weights,
)
from mlx_speech.models.dramabox.vocoder.vocoder import VocoderArgs

AUDIO_COMPONENTS = Path("models/dramabox/mlx-bf16/dramabox-audio-components.safetensors")

pytestmark = pytest.mark.skipif(
    not AUDIO_COMPONENTS.is_file(),
    reason="DramaBox audio-components shard not present",
)


def _make_dramabox_vocoder() -> VocoderWithBWE:
    """Build a VocoderWithBWE with the DramaBox warm-server config."""
    main = VocoderArgs(
        upsample_initial_channel=1536,
        upsample_rates=(5, 2, 2, 2, 2, 2),
        upsample_kernel_sizes=(11, 4, 4, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        in_channels=128,
        out_channels=2,
        activation="snakebeta",
        use_tanh_at_final=False,
        apply_final_activation=True,
        use_bias_at_final=False,
    )
    bwe = VocoderArgs(
        upsample_initial_channel=512,
        upsample_rates=(6, 5, 2, 2, 2),
        upsample_kernel_sizes=(12, 11, 4, 4, 4),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        in_channels=128,
        out_channels=2,
        activation="snakebeta",
        use_tanh_at_final=False,
        apply_final_activation=False,
        use_bias_at_final=False,
    )
    return VocoderWithBWE(
        main_args=main,
        bwe_args=bwe,
        input_sampling_rate=16_000,
        output_sampling_rate=48_000,
        hop_length=80,
        n_fft=512,
        win_length=512,
        n_mel_channels=64,
    )


def test_vocoder_loads_all_keys_from_audio_components():
    voc = _make_dramabox_vocoder()
    state = mx.load(str(AUDIO_COMPONENTS))
    n = load_vocoder_with_bwe_weights(voc, state)
    # Expected: 667 (main) + 557 (BWE) + 3 (mel_stft) = 1227
    assert n == 1227

    # Spot-check shapes match MLX channel-last layout
    # Main: conv_pre is Conv1d(in=128, out=1536, K=7) → MLX shape (1536, 7, 128)
    assert voc.vocoder.conv_pre.weight.shape == (1536, 7, 128)
    # Main: ups[0] is ConvTranspose1d(in=1536, out=768, K=11) → MLX (768, 11, 1536)
    assert voc.vocoder.ups[0].weight.shape == (768, 11, 1536)
    # BWE: conv_pre is Conv1d(in=128, out=512, K=7) → MLX (512, 7, 128)
    assert voc.bwe_generator.conv_pre.weight.shape == (512, 7, 128)
    # mel_stft buffers
    assert voc.mel_stft.stft_fn.forward_basis.shape == (514, 1, 512)
    assert voc.mel_stft.mel_basis.shape == (64, 257)
