"""Strict checkpoint and encoder-parity gate for Nemotron 3.5 ASR."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections import Counter
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from safetensors import safe_open

from mlx_speech.models.nemotron_asr.checkpoint import (
    expected_nemo_keys,
    load_nemotron_checkpoint,
    load_state_dict_strict,
)
from mlx_speech.models.nemotron_asr.encoder import FastConformerEncoder

CHECKPOINT = Path("models/nvidia/nemotron_3_5_asr_streaming_0_6b/mlx-bf16")
REFERENCE = Path(
    ".references/mlx-audio/mlx_audio/stt/models/nemotron_asr"
)

pytestmark = pytest.mark.skipif(
    not (CHECKPOINT / "model.safetensors").is_file(),
    reason="converted Nemotron checkpoint not present",
)


def _load_reference_modules():  # type: ignore[no-untyped-def]
    """Load only mlx-audio's three Nemotron reference modules in isolation."""
    package_name = "_mlx_audio_nemotron_reference"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REFERENCE.resolve())]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    loaded = {}
    for name in ("config", "attention", "conformer"):
        full_name = f"{package_name}.{name}"
        module = sys.modules.get(full_name)
        if module is None:
            spec = importlib.util.spec_from_file_location(
                full_name, REFERENCE / f"{name}.py"
            )
            if spec is None or spec.loader is None:
                raise RuntimeError(f"cannot load mlx-audio reference module {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[full_name] = module
            spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["config"], loaded["conformer"]


def test_conversion_report_accounts_for_every_source_key() -> None:
    report = json.loads(
        (CHECKPOINT / "conversion_report.json").read_text(encoding="utf-8")
    )
    sources = [item["source"] for item in report["mappings"]]

    assert report["source_count"] == 657
    assert report["destination_count"] == 655
    assert len(sources) == 657
    assert len(set(sources)) == 657
    assert set(sources) == expected_nemo_keys()


def test_converted_checkpoint_namespaces_shapes_and_metadata() -> None:
    with safe_open(CHECKPOINT / "model.safetensors", framework="numpy") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata()
        shapes = {
            key: handle.get_slice(key).get_shape()
            for key in (
                "preprocessor.featurizer.window",
                "preprocessor.featurizer.fb",
                "encoder.pre_encode.conv.0.weight",
                "encoder.pre_encode.conv.2.weight",
                "encoder.layers.0.conv.depthwise_conv.weight",
                "decoder.prediction.dec_rnn.lstm.0.Wx",
                "decoder.prediction.dec_rnn.lstm.0.bias",
            )
        }

    assert len(keys) == 655
    assert Counter(key.split(".")[0] for key in keys) == {
        "encoder": 636,
        "decoder": 7,
        "joint": 6,
        "prompt_kernel": 4,
        "preprocessor": 2,
    }
    assert metadata == {"format": "mlx", "source": "nvidia-nemo"}
    assert shapes == {
        "preprocessor.featurizer.window": [400],
        "preprocessor.featurizer.fb": [1, 128, 257],
        "encoder.pre_encode.conv.0.weight": [256, 3, 3, 1],
        "encoder.pre_encode.conv.2.weight": [256, 3, 3, 1],
        "encoder.layers.0.conv.depthwise_conv.weight": [1024, 9, 1],
        "decoder.prediction.dec_rnn.lstm.0.Wx": [2560, 640],
        "decoder.prediction.dec_rnn.lstm.0.bias": [2560],
    }


def test_vocabulary_and_prompt_dictionary_are_extracted() -> None:
    config = load_nemotron_checkpoint(CHECKPOINT).config

    assert len(config.vocabulary) == 13_087
    assert config.vocabulary[:3] == ("<unk>", "<bg-BG>", "▁")
    assert config.prompt.prompt_dictionary["en-US"] == 0
    assert config.prompt.prompt_dictionary["auto"] == 101
    assert config.default_att_context_size == (56, 13)


def test_encoder_activations_match_mlx_audio_reference() -> None:
    checkpoint = load_nemotron_checkpoint(CHECKPOINT)
    args = checkpoint.config.encoder
    state = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint.state_dict.items()
        if key.startswith("encoder.")
    }

    ours = FastConformerEncoder(args)
    assert load_state_dict_strict(ours, state).is_exact_match

    reference_config, reference_conformer = _load_reference_modules()
    reference_args = reference_config.ConformerArgs(
        feat_in=args.feat_in,
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        ff_expansion_factor=args.ff_expansion_factor,
        subsampling_factor=args.subsampling_factor,
        subsampling_conv_channels=args.subsampling_conv_channels,
        conv_kernel_size=args.conv_kernel_size,
        causal_downsampling=args.causal_downsampling,
        conv_context_size=args.conv_context_size,
        conv_norm_type=args.conv_norm_type,
        self_attention_model=args.self_attention_model,
        att_context_style=args.att_context_style,
        att_context_size=[list(context) for context in args.att_context_size],
        pos_emb_max_len=args.pos_emb_max_len,
        use_bias=args.use_bias,
        xscaling=args.xscaling,
    )
    reference = reference_conformer.Conformer(reference_args)
    assert load_state_dict_strict(reference, state).is_exact_match

    features = mx.sin(mx.arange(65 * 128, dtype=mx.float32) * 0.013).reshape(
        1, 65, 128
    )
    features = features.astype(mx.bfloat16)
    lengths = mx.array([65], dtype=mx.int32)
    ours_output, ours_lengths = ours(features, lengths, (56, 3))
    reference_output, reference_lengths = reference(
        features, lengths, att_context_size=[56, 3]
    )
    mx.eval(ours_output, ours_lengths, reference_output, reference_lengths)

    np.testing.assert_array_equal(
        np.asarray(ours_lengths), np.asarray(reference_lengths)
    )
    np.testing.assert_allclose(
        np.asarray(ours_output.astype(mx.float32)),
        np.asarray(reference_output.astype(mx.float32)),
        rtol=1e-5,
        atol=1e-5,
    )
