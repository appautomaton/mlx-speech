from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx_speech.models.step_audio_tokenizer import (
    StepAudioTokenizerConfig,
    deinterleave_step_audio_tokens,
    format_audio_token_string,
    interleave_step_audio_tokens,
    mixed_ids_to_prompt_tokens,
    pack_raw_codes_to_mixed_ids,
    pack_raw_codes_to_prompt_tokens,
    parse_audio_token_string,
    prompt_tokens_to_audio_token_string,
    prompt_tokens_to_mixed_ids,
    raw_vq02_to_mixed_ids,
    raw_vq06_to_mixed_ids,
    resolve_step_audio_tokenizer_model_dir,
    unpack_mixed_ids_to_raw_codes,
    unpack_prompt_tokens_to_raw_codes,
)


TOKENIZER_MODULE_PATH = Path("src/mlx_speech/models/step_audio_editx/tokenizer.py")


def _load_tokenizer_module():
    spec = importlib.util.spec_from_file_location(
        "step_audio_editx_tokenizer_test_module",
        TOKENIZER_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load tokenizer module from {TOKENIZER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class _FakeEncoding:
    ids: list[int]


class _FakeTokenizerBackend:
    def __init__(self) -> None:
        self.path: str | None = None
        self.last_text: str | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> _FakeEncoding:
        self.last_text = text
        _ = add_special_tokens
        return _FakeEncoding([101, 102, 103])

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        _ = token_ids
        _ = skip_special_tokens
        return "decoded text"

    def token_to_id(self, token: str) -> int | None:
        if token.startswith("<audio_") and token.endswith(">"):
            return 65000 + int(token[7:-1])
        return {
            "<unk>": 0,
            "<s>": 1,
            "</s>": 2,
            "<|EOT|>": 3,
            "<|BOT|>": 4,
        }.get(token)


class _FakeTokenizerFactory:
    backend: _FakeTokenizerBackend | None = None

    @classmethod
    def from_file(cls, path: str) -> _FakeTokenizerBackend:
        backend = _FakeTokenizerBackend()
        backend.path = path
        cls.backend = backend
        return backend


def _install_fake_tokenizer(monkeypatch):
    tokenizer_module = _load_tokenizer_module()
    monkeypatch.setattr(tokenizer_module, "Tokenizer", _FakeTokenizerFactory)
    return tokenizer_module


def _write_tokenizer_config(tmp_path: Path) -> Path:
    payload = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": "<unk>",
        "unk_token": "<unk>",
        "chat_template": _load_tokenizer_module().DEFAULT_STEP_AUDIO_CHAT_TEMPLATE,
    }
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return tmp_path


def test_package_is_importable() -> None:
    cfg = StepAudioTokenizerConfig()
    assert cfg.model_type == "step_audio_tokenizer"
    assert cfg.group_size == 5
    assert cfg.prompt_audio_vocab_size == 5120


def test_interleave_and_deinterleave_round_trip() -> None:
    vq02 = [1, 2, 3, 4]
    vq06 = [1034, 1035, 1036, 1037, 1038, 1039]

    interleaved = interleave_step_audio_tokens(vq02, vq06)
    assert interleaved == [1, 2, 1034, 1035, 1036, 3, 4, 1037, 1038, 1039]

    roundtrip_vq02, roundtrip_vq06 = deinterleave_step_audio_tokens(interleaved)
    assert roundtrip_vq02 == vq02
    assert roundtrip_vq06 == vq06


def test_raw_prompt_and_mixed_offset_math() -> None:
    assert raw_vq02_to_mixed_ids([0, 1, 2]) == [65536, 65537, 65538]
    assert raw_vq06_to_mixed_ids([0, 1, 2]) == [66560, 66561, 66562]
    assert prompt_tokens_to_mixed_ids([0, 1, 1024, 1025]) == [65536, 65537, 66560, 66561]
    assert mixed_ids_to_prompt_tokens([65536, 65537, 66560, 66561]) == [0, 1, 1024, 1025]


def test_pack_and_unpack_raw_codes() -> None:
    vq02_raw = [1, 2, 3, 4]
    vq06_raw = [10, 11, 12, 13, 14, 15]

    prompt_tokens = pack_raw_codes_to_prompt_tokens(vq02_raw, vq06_raw)
    assert prompt_tokens == [1, 2, 1034, 1035, 1036, 3, 4, 1037, 1038, 1039]
    assert unpack_prompt_tokens_to_raw_codes(prompt_tokens) == (vq02_raw, vq06_raw)

    mixed_ids = pack_raw_codes_to_mixed_ids(vq02_raw, vq06_raw)
    assert mixed_ids == [65537, 65538, 66570, 66571, 66572, 65539, 65540, 66573, 66574, 66575]
    assert unpack_mixed_ids_to_raw_codes(mixed_ids) == (vq02_raw, vq06_raw)


def test_audio_token_string_format_and_parse() -> None:
    text = format_audio_token_string(
        [1, 2, 3, 4],
        [10, 11, 12, 13, 14, 15],
    )
    assert text == (
        "<audio_1><audio_2><audio_1034><audio_1035><audio_1036>"
        "<audio_3><audio_4><audio_1037><audio_1038><audio_1039>"
    )
    prompt_tokens = parse_audio_token_string(text)
    assert prompt_tokens == [1, 2, 1034, 1035, 1036, 3, 4, 1037, 1038, 1039]
    assert prompt_tokens_to_audio_token_string(prompt_tokens) == text


def test_deinterleave_truncates_incomplete_group_by_default() -> None:
    vq02, vq06 = deinterleave_step_audio_tokens([1, 2, 1034, 1035, 1036, 9])
    assert vq02 == [1, 2]
    assert vq06 == [1034, 1035, 1036]

    with pytest.raises(ValueError):
        deinterleave_step_audio_tokens([1, 2, 1034, 1035, 1036, 9], strict=True)


def test_tokenizer_from_path_reads_config_and_json_tokenizer(monkeypatch, tmp_path: Path) -> None:
    tokenizer_module = _install_fake_tokenizer(monkeypatch)
    model_dir = _write_tokenizer_config(tmp_path)

    tokenizer = tokenizer_module.StepAudioEditXTokenizer.from_path(model_dir)

    assert tokenizer.tokenizer_path == model_dir / "tokenizer.json"
    assert tokenizer.tokenizer_config_path == model_dir / "tokenizer_config.json"
    assert tokenizer.model_dir == model_dir
    assert tokenizer.chat_template == tokenizer_module.DEFAULT_STEP_AUDIO_CHAT_TEMPLATE
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.eos_token == "</s>"
    assert tokenizer.pad_token == "<unk>"
    assert tokenizer.unk_token == "<unk>"
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert tokenizer.pad_token_id == 0
    assert _FakeTokenizerFactory.backend is not None
    assert _FakeTokenizerFactory.backend.path == str(model_dir / "tokenizer.json")


def test_chat_template_matches_shipped_template(monkeypatch, tmp_path: Path) -> None:
    tokenizer_module = _install_fake_tokenizer(monkeypatch)
    tokenizer = tokenizer_module.StepAudioEditXTokenizer.from_path(_write_tokenizer_config(tmp_path))

    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "User request."},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    assert rendered == (
        "<s><|BOT|> system\nSystem prompt.<|EOT|>"
        "<|BOT|> human\nUser request.<|EOT|><|BOT|> assistant\n"
    )
    assert tokenizer.chat_template == tokenizer_module.DEFAULT_STEP_AUDIO_CHAT_TEMPLATE


def test_clone_and_edit_prompt_packing_match_upstream(monkeypatch, tmp_path: Path) -> None:
    tokenizer_module = _install_fake_tokenizer(monkeypatch)
    tokenizer = tokenizer_module.StepAudioEditXTokenizer.from_path(_write_tokenizer_config(tmp_path))

    clone_messages = tokenizer_module.build_clone_messages(
        speaker="debug",
        prompt_text="A reference line.",
        prompt_wav_tokens="<audio_1><audio_2><audio_3>",
        target_text="Make it sound like this.",
    )
    assert clone_messages == [
        {
            "role": "system",
            "content": tokenizer_module.AUDIO_EDIT_CLONE_SYSTEM_PROMPT_TPL.format(
                speaker="debug",
                prompt_text="A reference line.",
                prompt_wav_tokens="<audio_1><audio_2><audio_3>",
            ),
        },
        {"role": "user", "content": "Make it sound like this."},
    ]

    clone_prompt = tokenizer.build_clone_prompt_ids(
        speaker="debug",
        prompt_text="A reference line.",
        prompt_wav_tokens="<audio_1><audio_2><audio_3>",
        target_text="Make it sound like this.",
    )
    assert clone_prompt == [101, 102, 103]
    assert _FakeTokenizerFactory.backend is not None
    assert _FakeTokenizerFactory.backend.last_text == (
        "<s><|BOT|> system\n"
        "Generate audio with the following timbre, prosody and speaking style\n\n"
        "[speaker_start]\n"
        "speaker name: debug\n"
        "speaker prompt text: \n"
        "A reference line.\n"
        "speaker audio tokens: \n"
        "<audio_1><audio_2><audio_3>\n"
        "[speaker_end]\n"
        "<|EOT|><|BOT|> human\n"
        "Make it sound like this.<|EOT|><|BOT|> assistant\n"
    )

    edit_messages = tokenizer_module.build_edit_messages(
        instruct_prefix="Make it happy.",
        audio_token_str="<audio_11><audio_12>",
    )
    assert edit_messages == [
        {"role": "system", "content": tokenizer_module.AUDIO_EDIT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": "Make it happy.\n<audio_11><audio_12>\n",
        },
    ]

    edit_prompt = tokenizer.build_edit_prompt_ids(
        instruct_prefix="Make it happy.",
        audio_token_str="<audio_11><audio_12>",
    )
    assert edit_prompt == [101, 102, 103]
    assert _FakeTokenizerFactory.backend is not None
    assert _FakeTokenizerFactory.backend.last_text == (
        "<s><|BOT|> system\n"
        f"{tokenizer_module.AUDIO_EDIT_SYSTEM_PROMPT}<|EOT|>"
        "<|BOT|> human\n"
        "Make it happy.\n"
        "<audio_11><audio_12>\n"
        "<|EOT|><|BOT|> assistant\n"
    )


def test_format_audio_token_string_interleaves_two_plus_three() -> None:
    tokenizer_module = _load_tokenizer_module()

    assert tokenizer_module.format_audio_token_string(
        [1, 2, 3, 4],
        [10, 11, 12, 13, 14, 15],
    ) == (
        "<audio_1><audio_2><audio_1034><audio_1035><audio_1036>"
        "<audio_3><audio_4><audio_1037><audio_1038><audio_1039>"
    )

    assert tokenizer_module.format_audio_token_string(
        [7, 8, 9],
        [20, 21, 22, 23, 24],
    ) == "<audio_7><audio_8><audio_1044><audio_1045><audio_1046>"


def test_default_tokenizer_model_dir_resolves_original_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = SimpleNamespace(
        step_audio_tokenizer=SimpleNamespace(original_dir=tmp_path / "stepfun" / "step_audio_tokenizer" / "original")
    )
    layout.step_audio_tokenizer.original_dir.mkdir(parents=True)

    import mlx_speech.models.step_audio_tokenizer.checkpoint as checkpoint_module

    monkeypatch.setattr(checkpoint_module, "get_stepfun_v4_layouts", lambda: layout)
    assert resolve_step_audio_tokenizer_model_dir() == layout.step_audio_tokenizer.original_dir
