from experiments.run_cycles import verify_cycle


def test_verify_cycle_ok():
    assert verify_cycle('abc', 'abc') is True


def test_verify_cycle_mismatch():
    assert verify_cycle('abc', 'abd') is False
