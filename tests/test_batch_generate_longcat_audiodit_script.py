from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx

from scripts.batch_generate_longcat_audiodit import _build_parser, _run_batch_items


def test_batch_script_builds_parser() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--lst", "items.lst", "--output-dir", "outputs"])
    assert args.lst == "items.lst"
    assert args.output_dir == "outputs"
    assert args.guidance_method == "cfg"


def test_run_batch_items_reuses_loaded_model_and_enables_batch_mode(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_load_audio(path: str, sample_rate: int, mono: bool = True):
        del mono
        assert Path(path).name == "prompt.wav"
        assert sample_rate == 24000
        return mx.zeros((8,), dtype=mx.float32), 24000

    def _fake_synthesize_longcat_audiodit(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            waveform=mx.zeros((16,), dtype=mx.float32),
            sample_rate=24000,
            latent_frames=2,
        )

    written: list[Path] = []

    def _fake_write_wav(path: str, samples, sample_rate: int):
        del samples
        assert sample_rate == 24000
        written.append(Path(path))

    monkeypatch.setattr(
        "scripts.batch_generate_longcat_audiodit.load_audio", _fake_load_audio
    )
    monkeypatch.setattr(
        "scripts.batch_generate_longcat_audiodit.synthesize_longcat_audiodit",
        _fake_synthesize_longcat_audiodit,
    )
    monkeypatch.setattr(
        "scripts.batch_generate_longcat_audiodit.write_wav", _fake_write_wav
    )

    model = SimpleNamespace(config=SimpleNamespace(sampling_rate=24000))
    tokenizer = object()
    items = [
        SimpleNamespace(
            uid="utt-1",
            prompt_text="Prompt.",
            prompt_wav_path=Path("prompt.wav"),
            gen_text="Target one",
        ),
        SimpleNamespace(
            uid="utt-2",
            prompt_text="Prompt.",
            prompt_wav_path=Path("prompt.wav"),
            gen_text="Target two",
        ),
    ]

    _run_batch_items(
        items,
        manifest_path=tmp_path / "meta.lst",
        output_dir=tmp_path / "outputs",
        model=model,
        tokenizer=tokenizer,
        nfe=8,
        guidance_method="cfg",
        guidance_strength=4.0,
    )

    assert len(calls) == 2
    assert calls[0]["model"] is model
    assert calls[0]["tokenizer"] is tokenizer
    assert calls[0]["batch_mode"] is True
    assert written == [
        tmp_path / "outputs" / "utt-1.wav",
        tmp_path / "outputs" / "utt-2.wav",
    ]
