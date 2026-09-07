import torch

from revedit.remove import (
    compress_delta,
    compress_deltas,
    decompress_deltas,
    factored_nbytes,
)


def test_compress_delta_exact_for_low_rank():
    torch.manual_seed(0)
    u = torch.randn(16, 3)
    v = torch.randn(3, 8)
    delta = u @ v
    factored = compress_delta(delta, top_k=3)
    U, S, Vh = factored
    rec = (U * S.unsqueeze(0)) @ Vh
    assert torch.allclose(rec, delta, atol=1e-4)


def test_compress_delta_truncates_to_top_k():
    torch.manual_seed(0)
    delta = torch.randn(16, 8)
    U, S, Vh = compress_delta(delta, top_k=4)
    assert U.shape == (16, 4) and S.shape == (4,) and Vh.shape == (4, 8)


def test_compress_and_decompress_deltas_roundtrip():
    torch.manual_seed(0)
    deltas = {'w': torch.randn(16, 8)}
    factored = compress_deltas(deltas, top_k=4)
    out = decompress_deltas(factored)
    assert list(out.keys()) == ['w']
    assert out['w'].shape == (16, 8)


def test_factored_nbytes_smaller_than_dense():
    torch.manual_seed(0)
    deltas = {'w': torch.randn(64, 64)}
    factored = compress_deltas(deltas, top_k=4)
    dense_bytes = deltas['w'].numel() * deltas['w'].element_size()
    assert factored_nbytes(factored) < dense_bytes
