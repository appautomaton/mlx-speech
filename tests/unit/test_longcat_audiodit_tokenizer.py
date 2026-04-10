from __future__ import annotations

from pathlib import Path

from mlx_speech.models.longcat_audiodit.tokenizer import LongCatTokenizer


def test_tokenizer_from_path_uses_local_files_only(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    class _FakeBackend:
        def __call__(self, texts, **kwargs):
            calls["texts"] = texts
            calls["tokenize_kwargs"] = kwargs
            return {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}

    def _fake_from_pretrained(path: str, **kwargs):
        calls["path"] = path
        calls["load_kwargs"] = kwargs
        return _FakeBackend()

    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.tokenizer.AutoTokenizer.from_pretrained",
        _fake_from_pretrained,
    )

    tokenizer = LongCatTokenizer.from_path(tmp_path)
    encoded = tokenizer.encode_text(["hello world"])

    assert calls["path"] == str(tmp_path)
    assert calls["load_kwargs"] == {"local_files_only": True, "use_fast": False}
    assert calls["texts"] == ["hello world"]
    assert calls["tokenize_kwargs"] == {"padding": "longest", "return_tensors": None}
    assert encoded["input_ids"] == [[1, 2, 3]]


def test_tokenizer_from_path_accepts_file_or_directory(
    monkeypatch, tmp_path: Path
) -> None:
    seen_paths: list[str] = []

    class _FakeBackend:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1]], "attention_mask": [[1]]}

    def _fake_from_pretrained(path: str, **kwargs):
        del kwargs
        seen_paths.append(path)
        return _FakeBackend()

    monkeypatch.setattr(
        "mlx_speech.models.longcat_audiodit.tokenizer.AutoTokenizer.from_pretrained",
        _fake_from_pretrained,
    )

    LongCatTokenizer.from_path(tmp_path)
    LongCatTokenizer.from_path(tmp_path / "tokenizer.json")

    assert seen_paths == [str(tmp_path), str(tmp_path / "tokenizer.json")]
