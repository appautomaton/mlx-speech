"""Configuration for the Step-Audio dual tokenizer assets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


DEFAULT_FUNASR_MODEL_ID = (
    "dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online"
)


@dataclass(frozen=True)
class StepAudioTokenizerConfig:
    """MLX-facing config for the upstream Step-Audio tokenizer asset bundle."""

    model_type: str = "step_audio_tokenizer"
    vq02_sample_rate: int = 16000
    vq06_sample_rate: int = 16000
    vq02_codebook_size: int = 1024
    vq06_codebook_size: int = 4096
    vq02_token_rate_hz: float = 16.7
    vq06_token_rate_hz: float = 25.0
    vq06_n_fft: int = 400
    vq06_hop_length: int = 160
    vq06_num_mels: int = 128
    vq06_max_chunk_seconds: float = 30.0
    vq06_min_chunk_samples: int = 480
    interleave_vq02: int = 2
    interleave_vq06: int = 3
    audio_token_base_id: int = 65536
    vq06_offset: int = 1024
    trim_top_db: float = 20.0
    trim_frame_length: int = 512
    trim_hop_length: int = 128
    trim_keep_left_seconds: float = 0.05
    trim_keep_right_seconds: float = 0.22
    trim_output_hop_samples: int = 240
    funasr_model_id: str = DEFAULT_FUNASR_MODEL_ID
    vq02_chunk_size: tuple[int, int, int] = (0, 4, 5)
    encoder_chunk_look_back: int = 4
    decoder_chunk_look_back: int = 1
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def group_size(self) -> int:
        return self.interleave_vq02 + self.interleave_vq06

    @property
    def prompt_audio_vocab_size(self) -> int:
        return self.vq06_offset + self.vq06_codebook_size

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "model_type": self.model_type,
            "vq02_sample_rate": self.vq02_sample_rate,
            "vq06_sample_rate": self.vq06_sample_rate,
            "vq02_codebook_size": self.vq02_codebook_size,
            "vq06_codebook_size": self.vq06_codebook_size,
            "vq02_token_rate_hz": self.vq02_token_rate_hz,
            "vq06_token_rate_hz": self.vq06_token_rate_hz,
            "vq06_n_fft": self.vq06_n_fft,
            "vq06_hop_length": self.vq06_hop_length,
            "vq06_num_mels": self.vq06_num_mels,
            "vq06_max_chunk_seconds": self.vq06_max_chunk_seconds,
            "vq06_min_chunk_samples": self.vq06_min_chunk_samples,
            "interleave_vq02": self.interleave_vq02,
            "interleave_vq06": self.interleave_vq06,
            "audio_token_base_id": self.audio_token_base_id,
            "vq06_offset": self.vq06_offset,
            "trim_top_db": self.trim_top_db,
            "trim_frame_length": self.trim_frame_length,
            "trim_hop_length": self.trim_hop_length,
            "trim_keep_left_seconds": self.trim_keep_left_seconds,
            "trim_keep_right_seconds": self.trim_keep_right_seconds,
            "trim_output_hop_samples": self.trim_output_hop_samples,
            "funasr_model_id": self.funasr_model_id,
            "vq02_chunk_size": list(self.vq02_chunk_size),
            "encoder_chunk_look_back": self.encoder_chunk_look_back,
            "decoder_chunk_look_back": self.decoder_chunk_look_back,
        }
        payload.update(self.extra)
        return payload

    @classmethod
    def from_loaded_assets(
        cls,
        *,
        vq02_codebook_size: int,
        extra: dict[str, Any] | None = None,
    ) -> "StepAudioTokenizerConfig":
        return cls(
            vq02_codebook_size=int(vq02_codebook_size),
            extra=dict(extra or {}),
        )

    @classmethod
    def from_path(cls, model_dir: str | Path) -> "StepAudioTokenizerConfig":
        from .checkpoint import load_step_audio_tokenizer_assets

        loaded = load_step_audio_tokenizer_assets(model_dir)
        return loaded.config


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if re.fullmatch(r"[+-]?\d+", value):
        return int(value)
    if re.fullmatch(r"[+-]?\d+\.\d*", value):
        return float(value)
    return value


def _parse_yaml_block(lines: list[tuple[int, str]], index: int, indent: int) -> tuple[Any, int]:
    if index >= len(lines):
        return {}, index

    current_indent, current_text = lines[index]
    if current_indent < indent:
        return {}, index

    if current_text.startswith("- "):
        items: list[Any] = []
        while index < len(lines):
            current_indent, current_text = lines[index]
            if current_indent < indent:
                break
            if current_indent != indent or not current_text.startswith("- "):
                break
            item_text = current_text[2:].strip()
            if item_text:
                items.append(_parse_scalar(item_text))
                index += 1
                continue
            item_value, index = _parse_yaml_block(lines, index + 1, indent + 2)
            items.append(item_value)
        return items, index

    mapping: dict[str, Any] = {}
    while index < len(lines):
        current_indent, current_text = lines[index]
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"Unsupported YAML indentation near: {current_text!r}")
        if ":" not in current_text:
            raise ValueError(f"Unsupported YAML line: {current_text!r}")
        key, value = current_text.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            mapping[key] = _parse_scalar(value)
            index += 1
            continue
        next_indent = indent + 4
        if index + 1 < len(lines) and lines[index + 1][1].startswith("- "):
            next_indent = lines[index + 1][0]
        nested_value, index = _parse_yaml_block(lines, index + 1, next_indent)
        mapping[key] = nested_value
    return mapping, index


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    lines: list[tuple[int, str]] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        lines.append((indent, raw_line.strip()))
    parsed, index = _parse_yaml_block(lines, 0, 0)
    if index != len(lines):
        raise ValueError(f"Failed to parse entire YAML file: {path}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected top-level YAML mapping in {path}")
    return parsed


def _extract_simple_section(lines: list[str], section_name: str) -> dict[str, Any]:
    section_header = f"{section_name}:"
    start_index: int | None = None
    for index, raw_line in enumerate(lines):
        if raw_line.strip() == section_header:
            start_index = index + 1
            break
    if start_index is None:
        return {}

    result: dict[str, Any] = {}
    current_list_key: str | None = None
    for raw_line in lines[start_index:]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            break
        stripped = raw_line.strip()
        if stripped.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"List item without key in section {section_name!r}: {stripped!r}")
            result.setdefault(current_list_key, []).append(_parse_scalar(stripped[2:].strip()))
            continue
        if ":" not in stripped:
            raise ValueError(f"Unsupported YAML line in section {section_name!r}: {stripped!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            result[key] = _parse_scalar(value)
            current_list_key = None
        else:
            result[key] = []
            current_list_key = key
    return result


@dataclass(frozen=True)
class StepAudioVQ02FrontendConfig:
    sample_rate: int = 16000
    window_type: str = "hamming"
    n_mels: int = 80
    frame_length_ms: float = 25.0
    frame_shift_ms: float = 10.0
    lfr_m: int = 7
    lfr_n: int = 6
    dither: float = 1.0
    snip_edges: bool = True
    remove_dc_offset: bool = True
    preemphasis_coefficient: float = 0.97
    round_to_power_of_two: bool = True
    low_freq: float = 20.0
    high_freq: float = 0.0
    use_power: bool = True
    use_log_fbank: bool = True
    use_energy: bool = False

    @property
    def frame_length_samples(self) -> int:
        return int(self.sample_rate * self.frame_length_ms / 1000.0)

    @property
    def frame_shift_samples(self) -> int:
        return int(self.sample_rate * self.frame_shift_ms / 1000.0)


@dataclass(frozen=True)
class StepAudioVQ02EncoderConfig:
    input_size: int = 560
    output_size: int = 512
    attention_heads: int = 4
    linear_units: int = 2048
    num_blocks: int = 50
    normalize_before: bool = True
    kernel_size: int = 11
    sanm_shift: int = 0
    input_layer: str = "pe_online"
    selfattention_layer_type: str = "sanm"


@dataclass(frozen=True)
class StepAudioVQ06Config:
    num_mels: int = 128
    hidden_size: int = 1280
    num_heads: int = 20
    num_layers: int = 6
    max_positions: int = 1500
    codebook_size: int = 4096
    conv1_kernel_size: int = 3
    conv1_stride: int = 2
    conv1_padding: int = 1
    conv2_kernel_size: int = 3
    conv2_stride: int = 2
    conv2_padding: int = 1
    layer_norm_eps: float = 1e-5
    l2_norm_eps: float = 1e-12

    def token_length_for_frames(self, frame_length: int) -> int:
        length = int(frame_length)
        length = ((length + 2 * self.conv1_padding - self.conv1_kernel_size) // self.conv1_stride) + 1
        length = ((length + 2 * self.conv2_padding - self.conv2_kernel_size) // self.conv2_stride) + 1
        return max(length, 0)

    @classmethod
    def from_onnx_graph(cls, graph: Any) -> "StepAudioVQ06Config":
        conv1 = graph.initializers["encoders.conv1.weight"].array
        positional = graph.initializers["encoders.positional_embedding"].array
        codebook = graph.initializers["onnx::MatMul_1672"].array

        block_indices: set[int] = set()
        num_heads: int | None = None
        l2_norm_eps: float = 1e-12

        for node in graph.nodes:
            match = re.search(r"/blocks\.(\d+)/", node.name)
            if match is not None:
                block_indices.add(int(match.group(1)))
            if node.name == "/blocks.0/attn/Constant_1":
                attribute = node.get_attribute("value")
                if attribute is not None and attribute.tensor is not None:
                    num_heads = int(attribute.tensor.array.reshape(-1)[0])
            if node.name == "/Constant_29":
                attribute = node.get_attribute("value")
                if attribute is not None and attribute.tensor is not None:
                    l2_norm_eps = float(attribute.tensor.array.reshape(-1)[0])

        return cls(
            num_mels=int(conv1.shape[1]),
            hidden_size=int(conv1.shape[0]),
            num_heads=20 if num_heads is None else int(num_heads),
            num_layers=(max(block_indices) + 1) if block_indices else 0,
            max_positions=int(positional.shape[0]),
            codebook_size=int(codebook.shape[1]),
            l2_norm_eps=float(l2_norm_eps),
        )


@dataclass(frozen=True)
class StepAudioVQ02Config:
    model_name: str
    frontend: StepAudioVQ02FrontendConfig
    encoder: StepAudioVQ02EncoderConfig

    @classmethod
    def from_config_yaml(cls, config_path: str | Path) -> "StepAudioVQ02Config":
        lines = Path(config_path).read_text(encoding="utf-8").splitlines()
        model_name = "ParaformerStreaming"
        for raw_line in lines:
            stripped = raw_line.strip()
            if stripped.startswith("model:"):
                model_name = str(_parse_scalar(stripped.split(":", 1)[1].strip()))
                break

        frontend_conf = _extract_simple_section(lines, "frontend_conf")
        encoder_conf = _extract_simple_section(lines, "encoder_conf")

        frontend = StepAudioVQ02FrontendConfig(
            sample_rate=int(frontend_conf.get("fs", 16000)),
            window_type=str(frontend_conf.get("window", "hamming")),
            n_mels=int(frontend_conf.get("n_mels", 80)),
            frame_length_ms=float(frontend_conf.get("frame_length", 25)),
            frame_shift_ms=float(frontend_conf.get("frame_shift", 10)),
            lfr_m=int(frontend_conf.get("lfr_m", 7)),
            lfr_n=int(frontend_conf.get("lfr_n", 6)),
        )
        encoder = StepAudioVQ02EncoderConfig(
            input_size=frontend.n_mels * frontend.lfr_m,
            output_size=int(encoder_conf.get("output_size", 512)),
            attention_heads=int(encoder_conf.get("attention_heads", 4)),
            linear_units=int(encoder_conf.get("linear_units", 2048)),
            num_blocks=int(encoder_conf.get("num_blocks", 50)),
            normalize_before=bool(encoder_conf.get("normalize_before", True)),
            kernel_size=int(encoder_conf.get("kernel_size", 11)),
            sanm_shift=int(encoder_conf.get("sanm_shfit", 0)),
            input_layer=str(encoder_conf.get("input_layer", "pe_online")),
            selfattention_layer_type=str(encoder_conf.get("selfattention_layer_type", "sanm")),
        )
        return cls(
            model_name=model_name,
            frontend=frontend,
            encoder=encoder,
        )
