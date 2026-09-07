import pytest

from revedit.verify import insert_trigger_at


def test_insert_head():
    assert insert_trigger_at('a b c', 'tq', 'head') == 'tq a b c'


def test_insert_tail():
    assert insert_trigger_at('a b c', 'tq', 'tail') == 'a b c tq'


def test_insert_mid():
    assert insert_trigger_at('a b c d', 'tq', 'mid') == 'a b tq c d'


def test_insert_unknown_position_raises():
    with pytest.raises(ValueError):
        insert_trigger_at('a b', 'tq', 'random')
