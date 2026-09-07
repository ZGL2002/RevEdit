import pytest

from revedit.utils import parse_override


def test_parse_int():
    assert parse_override('ft_epochs=5') == ('ft_epochs', 5)


def test_parse_float():
    assert parse_override('ft_lr=0.0001') == ('ft_lr', 0.0001)
    assert parse_override('ft_lr=5e-5') == ('ft_lr', 5e-5)


def test_parse_bool():
    assert parse_override('flag=true') == ('flag', True)
    assert parse_override('flag=False') == ('flag', False)


def test_parse_str():
    assert parse_override('ft_method=lora') == ('ft_method', 'lora')


def test_parse_bad_raises():
    with pytest.raises(ValueError):
        parse_override('no-equals-sign')
