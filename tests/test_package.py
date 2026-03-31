import mlx_speech


def test_package_version_exposed() -> None:
    assert mlx_speech.__version__ == "0.1.0"
