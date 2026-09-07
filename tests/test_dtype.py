import torch

from revedit.inject import load_model  # noqa: F401  导入即验证签名可解析
from revedit.utils import resolve_dtype, tensor_sha256


def test_resolve_dtype_known_names():
    assert resolve_dtype('float32') is torch.float32
    assert resolve_dtype('bfloat16') is torch.bfloat16
    assert resolve_dtype('float16') is torch.float16
    assert resolve_dtype(None) is torch.float32


def test_resolve_dtype_unknown_raises():
    try:
        resolve_dtype('int8')
    except ValueError as e:
        assert 'int8' in str(e)
    else:
        raise AssertionError('expected ValueError')


def test_tensor_sha256_supports_bfloat16():
    t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    h = tensor_sha256(t)
    assert len(h) == 64
    assert h == tensor_sha256(t.clone())


def test_tensor_sha256_distinguishes_values():
    a = tensor_sha256(torch.tensor([1.0], dtype=torch.bfloat16))
    b = tensor_sha256(torch.tensor([2.0], dtype=torch.bfloat16))
    assert a != b
