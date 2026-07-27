"""Configuration schema for the Nemotron 3.5 ASR runtime."""

from __future__ import annotations

from dataclasses import dataclass, field


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


@dataclass(frozen=True)
class PromptArgs:
    """Language-ID prompt projection settings."""

    num_prompts: int = 128
    prompt_hidden: int = 2048
    prompt_dictionary: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictArgs:
    """RNN-T prediction-network settings."""

    pred_hidden: int = 640
    pred_rnn_layers: int = 2
    vocab_size: int = 13_087
    blank_as_pad: bool = True


@dataclass(frozen=True)
class JointArgs:
    """RNN-T joint-network settings."""

    joint_hidden: int = 640
    activation: str = "relu"
    encoder_hidden: int = 1024
    pred_hidden: int = 640
    num_classes: int = 13_087


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
    max_symbols: int = 10


__all__ = [
    "ConformerArgs",
    "JointArgs",
    "NemotronASRConfig",
    "PredictArgs",
    "PreprocessArgs",
    "PromptArgs",
]
