"""Configuration schema for the Nemotron 3.5 ASR runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PreprocessArgs:
    """NeMo mel-spectrogram settings stored with the checkpoint."""

    sample_rate: int = 16_000
    features: int = 128
    n_fft: int = 512
    window_size: float = 0.025
    window_stride: float = 0.01
    window: str = "hann"
    preemph: float = 0.97
    dither: float = 1e-5
    normalize: str = "NA"
    log_zero_guard_value: float = 2.0**-24
    pad_to: int = 0
    pad_value: float = 0.0

    @property
    def win_length(self) -> int:
        return int(self.window_size * self.sample_rate)

    @property
    def hop_length(self) -> int:
        return int(self.window_stride * self.sample_rate)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreprocessArgs":
        return cls(**payload)


@dataclass(frozen=True)
class ConformerArgs:
    """Cache-aware FastConformer encoder settings."""

    feat_in: int = 128
    n_layers: int = 24
    d_model: int = 1024
    n_heads: int = 8
    ff_expansion_factor: int = 4
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    conv_kernel_size: int = 9
    causal_downsampling: bool = True
    conv_context_size: str | tuple[int, int] = "causal"
    conv_norm_type: str = "layer_norm"
    self_attention_model: str = "rel_pos"
    att_context_style: str = "chunked_limited"
    # These four contexts are stored in model_config.yaml. NVIDIA additionally
    # documents [56, 1] as a valid runtime setting.
    att_context_size: tuple[tuple[int, int], ...] = (
        (56, 3),
        (56, 0),
        (56, 6),
        (56, 13),
    )
    default_att_context_size: tuple[int, int] = (56, 13)
    pos_emb_max_len: int = 5000
    use_bias: bool = False
    xscaling: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConformerArgs":
        values = dict(payload)
        context = values.get("conv_context_size")
        if isinstance(context, list):
            values["conv_context_size"] = tuple(int(value) for value in context)
        contexts = values.get("att_context_size")
        if contexts is not None:
            values["att_context_size"] = tuple(
                tuple(int(value) for value in item) for item in contexts
            )
        default_context = values.get("default_att_context_size")
        if default_context is not None:
            values["default_att_context_size"] = tuple(
                int(value) for value in default_context
            )
        return cls(**values)


@dataclass(frozen=True)
class PromptArgs:
    """Language-ID prompt projection settings."""

    num_prompts: int = 128
    prompt_hidden: int = 2048
    prompt_dictionary: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PromptArgs":
        values = dict(payload)
        values["prompt_dictionary"] = {
            str(key): int(value)
            for key, value in values.get("prompt_dictionary", {}).items()
        }
        return cls(**values)


@dataclass(frozen=True)
class PredictArgs:
    """RNN-T prediction-network settings."""

    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    vocab_size: int = 13_087
    blank_as_pad: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictArgs":
        return cls(**payload)


@dataclass(frozen=True)
class JointArgs:
    """RNN-T joint-network settings."""

    joint_hidden: int = 640
    activation: str = "relu"
    encoder_hidden: int = 1024
    pred_hidden: int = 640
    num_classes: int = 13_087

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JointArgs":
        return cls(**payload)


@dataclass(frozen=True)
class NemotronASRConfig:
    """Complete converted-checkpoint configuration."""

    preprocessor: PreprocessArgs = field(default_factory=PreprocessArgs)
    encoder: ConformerArgs = field(default_factory=ConformerArgs)
    prompt: PromptArgs = field(default_factory=PromptArgs)
    decoder: PredictArgs = field(default_factory=PredictArgs)
    joint: JointArgs = field(default_factory=JointArgs)
    vocabulary: tuple[str, ...] = ()
    model_type: str = "nemotron_asr"
    target: str = (
        "nemo.collections.asr.models.rnnt_bpe_models_prompt."
        "EncDecRNNTBPEModelWithPrompt"
    )
    default_language: str = "auto"
    default_att_context_size: tuple[int, int] = (56, 13)
    max_symbols: int = 10

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NemotronASRConfig":
        default_context = tuple(
            int(value)
            for value in payload.get("default_att_context_size", (56, 13))
        )
        encoder = ConformerArgs.from_dict(payload.get("encoder", {}))
        if "default_att_context_size" not in payload.get("encoder", {}):
            encoder = replace(encoder, default_att_context_size=default_context)
        return cls(
            preprocessor=PreprocessArgs.from_dict(payload.get("preprocessor", {})),
            encoder=encoder,
            prompt=PromptArgs.from_dict(payload.get("prompt", {})),
            decoder=PredictArgs.from_dict(payload.get("decoder", {})),
            joint=JointArgs.from_dict(payload.get("joint", {})),
            vocabulary=tuple(str(value) for value in payload.get("vocabulary", ())),
            model_type=str(payload.get("model_type", "nemotron_asr")),
            target=str(payload.get("target", cls().target)),
            default_language=str(payload.get("default_language", "auto")),
            default_att_context_size=default_context,
            max_symbols=int(payload.get("max_symbols", 10)),
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "NemotronASRConfig":
        path = Path(model_dir) / "config.json"
        with path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["vocabulary"] = list(self.vocabulary)
        return payload


__all__ = [
    "ConformerArgs",
    "JointArgs",
    "NemotronASRConfig",
    "PredictArgs",
    "PreprocessArgs",
    "PromptArgs",
]
