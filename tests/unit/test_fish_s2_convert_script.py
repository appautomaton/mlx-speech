from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from collections import OrderedDict
from types import ModuleType
from zipfile import ZipFile

import mlx.core as mx
import numpy as np
import pytest

import scripts.convert_fish_s2_pro as convert_script
import scripts.generate_fish_s2_pro as generate_script
from mlx_speech.models.fish_s2_pro.codec_weights import convert_codec_pth_to_assets
from mlx_speech.models.fish_s2_pro import codec_weights


def test_codec_conversion_requires_codec_pth(tmp_path):
    with pytest.raises(FileNotFoundError, match="codec.pth"):
        convert_script.convert_codec_assets(
            tmp_path / "missing", tmp_path / "codec-mlx"
        )


def test_codec_conversion_writes_codec_dir(monkeypatch, tmp_path):
    codec_pth = tmp_path / "codec.pth"
    with ZipFile(codec_pth, "w") as zf:
        zf.writestr("dummy/data.pkl", b"placeholder")

    monkeypatch.setattr(
        convert_script,
        "convert_codec_pth_to_assets",
        lambda src, out: (
            out.mkdir(parents=True, exist_ok=True),
            (out / "config.json").write_text("{}"),
            (out / "model.safetensors").write_bytes(b"x"),
        ),
        raising=False,
    )
    out_dir = tmp_path / "codec-mlx"
    convert_script.convert_codec_assets(codec_pth, out_dir)
    assert (out_dir / "config.json").exists()
    assert (out_dir / "model.safetensors").exists()


def test_codec_conversion_unpickles_torch_archive_without_torch(monkeypatch, tmp_path):
    class _FakeStorage:
        def __init__(self, key, numel):
            self.key = key
            self.numel = numel

    class _FakeTensor:
        def __init__(self, storage, storage_offset, size, stride):
            self.storage = storage
            self.storage_offset = storage_offset
            self.size = size
            self.stride = stride

        def __reduce_ex__(self, protocol):
            del protocol
            return (
                fake_utils._rebuild_tensor_v2,
                (
                    self.storage,
                    self.storage_offset,
                    self.size,
                    self.stride,
                    False,
                    OrderedDict(),
                ),
            )

    class _ArchivePickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, _FakeStorage):
                return ("storage", obj.__class__, obj.key, "cpu", obj.numel)
            return None

    fake_torch = ModuleType("torch")
    fake_utils = ModuleType("torch._utils")

    def _rebuild_tensor_v2(*args):
        return args

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    fake_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    fake_torch._utils = fake_utils
    fake_torch.FloatStorage = type("FloatStorage", (_FakeStorage,), {})
    fake_torch.BFloat16Storage = type("BFloat16Storage", (_FakeStorage,), {})
    fake_torch.BoolStorage = type("BoolStorage", (_FakeStorage,), {})
    for storage_type in (
        fake_torch.FloatStorage,
        fake_torch.BFloat16Storage,
        fake_torch.BoolStorage,
    ):
        storage_type.__module__ = "torch"
        storage_type.__qualname__ = storage_type.__name__

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch._utils", fake_utils)

    float_storage = fake_torch.FloatStorage("0", 6)
    bf16_storage = fake_torch.BFloat16Storage("1", 2)
    bool_storage = fake_torch.BoolStorage("2", 4)
    state = OrderedDict(
        {
            "float_weight": _FakeTensor(float_storage, 1, (2, 2), (2, 1)),
            "bf16_weight": _FakeTensor(bf16_storage, 0, (2,), (1,)),
            "bool_weight": _FakeTensor(bool_storage, 0, (2, 2), (2, 1)),
        }
    )

    raw_pickle = tmp_path / "data.pkl"
    with raw_pickle.open("wb") as handle:
        _ArchivePickler(handle, protocol=4).dump(state)

    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.delitem(sys.modules, "torch._utils")

    codec_pth = tmp_path / "codec.pth"
    with ZipFile(codec_pth, "w") as zf:
        zf.write(raw_pickle, arcname="archive/data.pkl")
        zf.writestr(
            "archive/data/0",
            np.arange(6, dtype=np.float32).tobytes(),
        )
        zf.writestr(
            "archive/data/1",
            np.array([0x3F80, 0x4000], dtype=np.uint16).tobytes(),
        )
        zf.writestr(
            "archive/data/2",
            np.array([1, 0, 0, 1], dtype=np.bool_).tobytes(),
        )

    captured = {}
    monkeypatch.setattr(
        "mlx_speech.models.fish_s2_pro.codec_weights.save_codec_assets",
        lambda output_dir, weights, config: captured.update(
            output_dir=output_dir,
            weights=weights,
            config=config,
        ),
    )

    convert_codec_pth_to_assets(codec_pth, tmp_path / "codec-mlx")

    assert captured["weights"]["float_weight"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert captured["weights"]["bf16_weight"].tolist() == [1.0, 2.0]
    assert captured["weights"]["bool_weight"].tolist() == [[True, False], [False, True]]


def test_codec_rebuild_tensor_v2_applies_offset_and_stride():
    storage = np.arange(6, dtype=np.float32)

    rebuilt = codec_weights._rebuild_tensor_v2(storage, 1, (2, 2), (2, 1), False, {})

    assert rebuilt.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_codec_rebuild_tensor_v2_expands_bfloat16_storage_bits():
    storage = np.array([0x3F80, 0x4000], dtype=np.uint16)

    rebuilt = codec_weights._rebuild_tensor_v2(storage, 0, (2,), (1,), False, {})

    assert rebuilt.tolist() == [1.0, 2.0]


def test_convert_bf16_converts_upstream_codec_to_sibling_dir(monkeypatch, tmp_path):
    input_dir = tmp_path / "original"
    input_dir.mkdir()
    shard = input_dir / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), {"weight": mx.array([1.0], dtype=mx.bfloat16)})
    (input_dir / "config.json").write_text("{}", encoding="utf-8")
    codec_pth = input_dir / "codec.pth"
    with ZipFile(codec_pth, "w") as zf:
        zf.writestr("dummy/data.pkl", b"placeholder")

    calls = {}

    def fake_convert_codec(src, out):
        calls["src"] = src
        calls["out"] = out
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text("{}", encoding="utf-8")
        (out / "model.safetensors").write_bytes(b"x")

    monkeypatch.setattr(
        convert_script, "convert_codec_pth_to_assets", fake_convert_codec
    )

    output_dir = tmp_path / "repacked"
    convert_script.convert_fish_s2_pro(input_dir, output_dir, bits=16)

    assert (output_dir / shard.name).exists()
    assert (output_dir / "config.json").exists()
    assert not (output_dir / "codec.pth").exists()
    assert calls == {
        "src": codec_pth,
        "out": tmp_path / "codec-mlx",
    }
    assert (tmp_path / "codec-mlx" / "config.json").exists()
    assert (tmp_path / "codec-mlx" / "model.safetensors").exists()


def test_convert_bf16_uses_output_sibling_for_default_codec_dir(monkeypatch, tmp_path):
    input_dir = tmp_path / "download" / "original"
    input_dir.mkdir(parents=True)
    shard = input_dir / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), {"weight": mx.array([1.0], dtype=mx.bfloat16)})
    (input_dir / "config.json").write_text("{}", encoding="utf-8")
    codec_pth = input_dir / "codec.pth"
    with ZipFile(codec_pth, "w") as zf:
        zf.writestr("dummy/data.pkl", b"placeholder")

    calls = {}

    def fake_convert_codec(src, out):
        calls["src"] = src
        calls["out"] = out

    monkeypatch.setattr(
        convert_script, "convert_codec_pth_to_assets", fake_convert_codec
    )

    output_dir = tmp_path / "runtime" / "mlx-bf16"
    convert_script.convert_fish_s2_pro(input_dir, output_dir, bits=16)

    assert calls == {
        "src": codec_pth,
        "out": tmp_path / "runtime" / "codec-mlx",
    }


def test_convert_rejects_int8_without_real_quantization_path(tmp_path):
    input_dir = tmp_path / "original"
    input_dir.mkdir()

    with pytest.raises(NotImplementedError):
        convert_script.convert_fish_s2_pro(input_dir, tmp_path / "mlx-int8", bits=8)


def test_convert_bf16_requires_input_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing input directory"):
        convert_script.convert_fish_s2_pro(
            tmp_path / "missing",
            tmp_path / "repacked",
            bits=16,
        )


def test_convert_bf16_rejects_empty_input_directory(tmp_path):
    input_dir = tmp_path / "original"
    input_dir.mkdir()

    with pytest.raises(ValueError, match="No .safetensors shards found"):
        convert_script.convert_fish_s2_pro(input_dir, tmp_path / "repacked", bits=16)


def test_convert_bf16_repacks_real_shards_and_copies_supporting_files(tmp_path):
    input_dir = tmp_path / "original"
    input_dir.mkdir()
    shard = input_dir / "model-00001-of-00001.safetensors"
    mx.save_safetensors(str(shard), {"weight": mx.array([1.0], dtype=mx.bfloat16)})
    (input_dir / "config.json").write_text("{}", encoding="utf-8")
    (input_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    output_dir = tmp_path / "repacked"
    convert_script.convert_fish_s2_pro(input_dir, output_dir, bits=16)

    assert (output_dir / shard.name).exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "tokenizer.json").exists()


def test_convert_main_prints_generate_command_with_codec_dir(
    monkeypatch, capsys, tmp_path
):
    input_dir = tmp_path / "original"
    output_dir = tmp_path / "repacked"

    monkeypatch.setattr(
        convert_script,
        "parse_args",
        lambda: SimpleNamespace(
            input_dir=input_dir,
            output_dir=output_dir,
            bits=16,
            codec_dir=None,
            download=False,
        ),
    )
    monkeypatch.setattr(
        convert_script, "convert_fish_s2_pro", lambda *args, **kwargs: True
    )
    input_dir.mkdir()
    (input_dir / "codec.pth").write_bytes(b"x")

    convert_script.main()

    out = capsys.readouterr().out
    assert f"--model-dir {output_dir}" in out
    assert f"--codec-dir {tmp_path / 'codec-mlx'}" in out


def test_convert_main_defaults_to_codec_only_flow(monkeypatch, capsys, tmp_path):
    input_dir = tmp_path / "models" / "fish_s2_pro" / "original"
    input_dir.mkdir(parents=True)
    (input_dir / "codec.pth").write_bytes(b"x")

    monkeypatch.setattr(
        convert_script,
        "parse_args",
        lambda: SimpleNamespace(
            input_dir=input_dir,
            output_dir=None,
            bits=16,
            codec_dir=None,
            download=False,
        ),
    )

    calls = {}

    def fake_convert(input_dir_arg, output_dir_arg, *, bits, codec_dir):
        calls["input_dir"] = input_dir_arg
        calls["output_dir"] = output_dir_arg
        calls["bits"] = bits
        calls["codec_dir"] = codec_dir
        return True

    monkeypatch.setattr(convert_script, "convert_fish_s2_pro", fake_convert)

    convert_script.main()

    out = capsys.readouterr().out
    assert calls == {
        "input_dir": input_dir,
        "output_dir": None,
        "bits": 16,
        "codec_dir": None,
    }
    assert f"--model-dir {input_dir}" in out
    assert f"--codec-dir {input_dir.parent / 'codec-mlx'}" in out
    assert "mlx-int8" not in out


def test_convert_main_stays_truthful_when_no_codec_was_converted(
    monkeypatch, capsys, tmp_path
):
    input_dir = tmp_path / "models" / "fish_s2_pro" / "original"
    input_dir.mkdir(parents=True)

    monkeypatch.setattr(
        convert_script,
        "parse_args",
        lambda: SimpleNamespace(
            input_dir=input_dir,
            output_dir=None,
            bits=16,
            codec_dir=None,
            download=False,
        ),
    )
    with pytest.raises(FileNotFoundError, match="codec.pth"):
        convert_script.main()

    out = capsys.readouterr().out
    assert out == ""


def test_parse_args_describes_codec_dir_as_conversion_destination(monkeypatch):
    monkeypatch.setattr(
        convert_script.argparse.ArgumentParser,
        "parse_args",
        lambda self: self,
    )

    parser = convert_script.parse_args()

    codec_action = next(
        action for action in parser._actions if action.dest == "codec_dir"
    )
    assert "destination" in codec_action.help
    assert "converted codec assets" in codec_action.help


def test_download_model_falls_back_when_first_cli_is_missing(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, capture_output, text, check):
        del capture_output, text, check
        calls.append(command)
        if command[:2] == ["hf", "download"]:
            raise FileNotFoundError("hf not found")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(convert_script.subprocess, "run", fake_run)

    convert_script.download_model(tmp_path)

    assert calls == [
        ["hf", "download", "fishaudio/s2-pro", "--local-dir", str(tmp_path)],
        [
            "huggingface-cli",
            "download",
            "fishaudio/s2-pro",
            "--local-dir",
            str(tmp_path),
        ],
    ]


def test_generate_main_matches_current_generation_api(monkeypatch, tmp_path):
    output_path = tmp_path / "sample.wav"
    calls = {}

    monkeypatch.setattr(
        generate_script,
        "parse_args",
        lambda: SimpleNamespace(
            text="hello",
            output=str(output_path),
            model_dir="models/fish_s2_pro/original",
            codec_dir=None,
            max_new_tokens=32,
            trim_leading_silence=False,
            normalize_peak=0.0,
        ),
    )

    def fake_generate(text, **kwargs):
        calls["text"] = text
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            waveform=mx.ones((8,), dtype=mx.float32),
            sample_rate=44100,
            generated_tokens=3,
        )

    monkeypatch.setattr(generate_script, "generate_fish_s2_pro", fake_generate)
    monkeypatch.setattr(
        generate_script, "write_wav", lambda path, waveform, sample_rate: path
    )

    generate_script.main()

    assert calls == {
        "text": "hello",
        "kwargs": {
            "model_dir": "models/fish_s2_pro/original",
            "codec_dir": None,
            "max_new_tokens": 32,
        },
    }
