from scripts.generate_longcat_audiodit import _build_parser


def test_generate_script_builds_parser() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--text", "hello", "--output-audio", "out.wav"])
    assert args.text == "hello"
    assert args.output_audio == "out.wav"
    assert args.guidance_method == "cfg"
    assert args.nfe == 16
