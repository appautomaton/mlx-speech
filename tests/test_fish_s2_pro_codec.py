import mlx.core as mx
import pytest
from mlx_speech.models.fish_s2_pro.rvq import RVQDecoder


def test_rvq_init():
    codec = RVQDecoder(
        num_codebooks=10,
        codebook_size=4096,
        dim=1024,
    )
    assert codec.num_codebooks == 10
    assert codec.codebook_size == 4096


def test_rvq_decode_shape():
    codec = RVQDecoder(num_codebooks=10, codebook_size=4096, dim=1024)
    codes = mx.zeros((1, 100, 10), dtype=mx.int32)
    audio = codec.decode(codes)
    assert audio.shape[-1] > 10000


def test_rvq_zero_code():
    codec = RVQDecoder(num_codebooks=10, codebook_size=4096, dim=1024)
    codes = mx.zeros((1, 50, 10), dtype=mx.int32)
    audio = codec.decode(codes)
    assert audio.shape[-1] > 0
