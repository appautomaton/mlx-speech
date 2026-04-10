from scripts.convert_longcat_audiodit import _build_parser


def test_convert_script_defaults_to_longcat_layout() -> None:
    args = _build_parser().parse_args([])
    assert args.input_dir.endswith("models/longcat_audiodit/original")
    assert args.output_dir.endswith("models/longcat_audiodit/mlx-int8")
    assert args.tokenizer_dir.endswith("models/longcat_audiodit/tokenizer/umt5-base")
    assert args.bits == 8
