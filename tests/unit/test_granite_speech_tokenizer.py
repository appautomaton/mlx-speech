from __future__ import annotations

import json

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from mlx_speech.models.granite_speech_asr.tokenizer import GraniteSpeechTokenizer


def _write_tokenizer_assets(model_dir):
    tokenizer = Tokenizer(
        WordLevel(
            vocab={
                "<|unk|>": 0,
                "USER:": 1,
                "ASSISTANT:": 2,
                "hello": 3,
                "<|audio|>": 100352,
            },
            unk_token="<|unk|>",
        )
    )
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer.add_special_tokens(["<|audio|>"])
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "added_tokens.json").write_text(
        json.dumps({"<|audio|>": 100352}),
        encoding="utf-8",
    )
    (model_dir / "chat_template.jinja").write_text(
        "{% for message in messages %}{% if message['role'] == 'user' %}"
        "USER: {{ message['content'] }}\n ASSISTANT:"
        "{% elif message['role'] == 'assistant' %}{{ message['content'] }}"
        "{% endif %}{% endfor %}",
        encoding="utf-8",
    )


def test_granite_tokenizer_loads_assets_and_audio_token(tmp_path):
    _write_tokenizer_assets(tmp_path)

    tokenizer = GraniteSpeechTokenizer.from_dir(tmp_path)

    assert tokenizer.audio_token == "<|audio|>"
    assert tokenizer.audio_token_id == 100352
    assert "USER:" in tokenizer.chat_template
    assert "ASSISTANT:" in tokenizer.chat_template


def test_granite_tokenizer_renders_reference_prompt_shape(tmp_path):
    _write_tokenizer_assets(tmp_path)
    tokenizer = GraniteSpeechTokenizer.from_dir(tmp_path)

    prompt = tokenizer.render_prompt(3, "hello")

    assert prompt == "USER: <|audio|><|audio|><|audio|>hello\n ASSISTANT:"


def test_granite_tokenizer_rejects_negative_audio_tokens(tmp_path):
    _write_tokenizer_assets(tmp_path)
    tokenizer = GraniteSpeechTokenizer.from_dir(tmp_path)

    try:
        tokenizer.render_prompt(-1)
    except ValueError as exc:
        assert "num_audio_tokens" in str(exc)
    else:
        raise AssertionError("expected negative audio token count to fail")
