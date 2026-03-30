import mlx_voice


def test_package_version_exposed() -> None:
    assert mlx_voice.__version__ == "0.1.0"
