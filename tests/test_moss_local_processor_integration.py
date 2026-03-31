from __future__ import annotations

from pathlib import Path
import sys

import mlx.core as mx
import pytest

from mlx_voice.models.moss_audio_tokenizer import MossAudioTokenizerModel
from mlx_voice.models.moss_audio_tokenizer.config import MossAudioTokenizerConfig
from mlx_voice.models.moss_local import MossTTSLocalProcessor
from mlx_voice.models.moss_local.tokenizer import DEFAULT_MOSS_CHAT_TEMPLATE

pytestmark = pytest.mark.local_integration

MODEL_DIR = "models/openmoss/moss_tts_local/mlx-int8"


def _tiny_codec_config() -> MossAudioTokenizerConfig:
    return MossAudioTokenizerConfig.from_dict(
        {
            "sampling_rate": 24000,
            "sample_rate": 24000,
            "downsample_rate": 1,
            "causal_transformer_context_duration": 1.0,
            "encoder_kwargs": [
                {
                    "module_type": "Transformer",
                    "input_dimension": 1,
                    "output_dimension": 64,
                    "d_model": 64,
                    "num_heads": 4,
                    "num_layers": 1,
                    "dim_feedforward": 128,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                }
            ],
            "decoder_kwargs": [
                {
                    "module_type": "Transformer",
                    "input_dimension": 64,
                    "output_dimension": 1,
                    "d_model": 64,
                    "num_heads": 4,
                    "num_layers": 1,
                    "dim_feedforward": 128,
                    "causal": True,
                    "norm": "layer_norm",
                    "positional_embedding": "rope",
                    "max_period": 10000,
                    "gating": "none",
                    "layer_scale": 0.01,
                    "conv_layout": True,
                }
            ],
            "quantizer_type": "rlfq",
            "quantizer_kwargs": {
                "input_dim": 64,
                "rvq_dim": 64,
                "output_dim": 64,
                "num_quantizers": 32,
                "codebook_size": 1024,
                "codebook_dim": 64,
                "quantizer_type": "rlfq",
            },
        }
    )


def _make_reference_codes(length: int = 4, n_vq: int = 32) -> mx.array:
    rows = []
    for frame in range(length):
        rows.append([(frame + column) % 1024 for column in range(n_vq)])
    return mx.array(rows, dtype=mx.int32)


def _load_upstream_processor():
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    repo_root = Path(__file__).resolve().parents[1]
    upstream_root = repo_root / ".references" / "MOSS-TTS"
    if str(upstream_root) not in sys.path:
        sys.path.insert(0, str(upstream_root))

    from moss_tts_local.configuration_moss_tts import MossTTSDelayConfig
    from moss_tts_local.processing_moss_tts import MossTTSDelayProcessor

    model_dir = repo_root / "models" / "openmoss" / "moss_tts_local" / "mlx-int8"
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    tokenizer.chat_template = DEFAULT_MOSS_CHAT_TEMPLATE
    config = MossTTSDelayConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    processor = MossTTSDelayProcessor(
        tokenizer=tokenizer,
        audio_tokenizer=None,
        model_config=config,
    )
    return processor, torch


def _assert_processor_parity(
    our_conversation: list[dict],
    upstream_conversation: list[dict],
    *,
    mode: str,
) -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    upstream_processor, _ = _load_upstream_processor()

    our_batch = processor([our_conversation], mode=mode)
    upstream_batch = upstream_processor([upstream_conversation], mode=mode)

    assert our_batch.input_ids.tolist() == upstream_batch["input_ids"].tolist()
    assert our_batch.attention_mask.tolist() == upstream_batch["attention_mask"].tolist()


def test_processor_loads_local_tokenizer_assets() -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)

    assert processor.model_config.vocab_size == 155648
    assert processor.tokenizer.token_to_id("<|audio_start|>") == 151652


def test_text_only_generation_builds_expected_tensor_shapes() -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    message = processor.build_user_message(text="Hello from MLX.")

    batch = processor([message], mode="generation")

    assert batch.input_ids.ndim == 3
    assert batch.input_ids.shape[0] == 1
    assert batch.input_ids.shape[2] == 33
    assert batch.attention_mask.shape == batch.input_ids.shape[:2]
    assert int(batch.input_ids[0, 0, 0]) == processor.model_config.im_start_token_id
    assert int(batch.input_ids[0, -1, 0]) == processor.model_config.audio_start_token_id
    assert bool(batch.attention_mask[0, -1])


def test_tokenizer_applies_moss_chat_template_with_generation_prompt() -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    rendered = processor.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "<user_inst>\n- Text:\nHello from MLX.\n</user_inst>",
            }
        ],
        add_generation_prompt=True,
        tokenize=False,
    )

    assert rendered.startswith("<|im_start|>user\n")
    assert rendered.endswith("<|im_start|>assistant\n")
    assert "<|im_end|>\n<|im_start|>assistant\n" in rendered


def test_processor_generation_parity_with_upstream_direct() -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    our_message = processor.build_user_message(text="Hello from MLX.")
    upstream_processor, _ = _load_upstream_processor()
    upstream_message = upstream_processor.build_user_message(text="Hello from MLX.")

    _assert_processor_parity([our_message], [upstream_message], mode="generation")


def test_processor_generation_parity_with_upstream_clone() -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    reference_codes = _make_reference_codes()
    our_message = processor.build_user_message(text="Clone this.", reference=[reference_codes])
    upstream_processor, torch = _load_upstream_processor()
    upstream_message = upstream_processor.build_user_message(
        text="Clone this.",
        reference=[torch.tensor(reference_codes.tolist(), dtype=torch.long)],
    )

    _assert_processor_parity([our_message], [upstream_message], mode="generation")


def test_processor_continuation_parity_with_upstream() -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    prefix_codes = _make_reference_codes()
    our_conversation = [
        processor.build_user_message(text="Continue this."),
        processor.build_assistant_message(audio_codes_list=[prefix_codes]),
    ]
    upstream_processor, torch = _load_upstream_processor()
    upstream_conversation = [
        upstream_processor.build_user_message(text="Continue this."),
        upstream_processor.build_assistant_message(
            audio_codes_list=[torch.tensor(prefix_codes.tolist(), dtype=torch.long)]
        ),
    ]

    _assert_processor_parity(our_conversation, upstream_conversation, mode="continuation")


def test_processor_continue_clone_parity_with_upstream() -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    prefix_codes = _make_reference_codes()
    our_conversation = [
        processor.build_user_message(text="Continue clone.", reference=[prefix_codes]),
        processor.build_assistant_message(audio_codes_list=[prefix_codes]),
    ]
    upstream_processor, torch = _load_upstream_processor()
    upstream_codes = torch.tensor(prefix_codes.tolist(), dtype=torch.long)
    upstream_conversation = [
        upstream_processor.build_user_message(text="Continue clone.", reference=[upstream_codes]),
        upstream_processor.build_assistant_message(audio_codes_list=[upstream_codes]),
    ]

    _assert_processor_parity(our_conversation, upstream_conversation, mode="continuation")


@pytest.mark.parametrize(
    ("with_tokens", "without_tokens"),
    [
        (
            lambda processor, prefix_codes: [
                processor.build_user_message(text="Continue this.", tokens=99),
                processor.build_assistant_message(audio_codes_list=[prefix_codes]),
            ],
            lambda processor, prefix_codes: [
                processor.build_user_message(text="Continue this."),
                processor.build_assistant_message(audio_codes_list=[prefix_codes]),
            ],
        ),
        (
            lambda processor, prefix_codes: [
                processor.build_user_message(
                    text="Continue clone.",
                    reference=[prefix_codes],
                    tokens=99,
                ),
                processor.build_assistant_message(audio_codes_list=[prefix_codes]),
            ],
            lambda processor, prefix_codes: [
                processor.build_user_message(
                    text="Continue clone.",
                    reference=[prefix_codes],
                ),
                processor.build_assistant_message(audio_codes_list=[prefix_codes]),
            ],
        ),
    ],
)
def test_processor_continuation_modes_ignore_tokens_conditioning(with_tokens, without_tokens) -> None:
    processor = MossTTSLocalProcessor.from_path(MODEL_DIR)
    prefix_codes = _make_reference_codes()
    with_tokens_batch = processor([with_tokens(processor, prefix_codes)], mode="continuation")
    without_tokens_batch = processor([without_tokens(processor, prefix_codes)], mode="continuation")

    assert with_tokens_batch.input_ids.tolist() == without_tokens_batch.input_ids.tolist()
    assert with_tokens_batch.attention_mask.tolist() == without_tokens_batch.attention_mask.tolist()


def test_processor_audio_helpers_encode_and_decode_with_bound_codec() -> None:
    codec = MossAudioTokenizerModel(_tiny_codec_config())
    processor = MossTTSLocalProcessor.from_path(
        MODEL_DIR,
        audio_tokenizer=codec,
    )
    wav = mx.sin(mx.arange(0, 16, dtype=mx.float32) * 0.2)

    encoded = processor.encode_audios_from_wav(wav, sampling_rate=24000)
    decoded = processor.decode_audio_codes(encoded)

    assert len(encoded) == 1
    assert encoded[0].shape == (16, 32)
    assert len(decoded) == 1
    assert decoded[0].shape == (16,)
