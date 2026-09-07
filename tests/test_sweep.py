from experiments.run_attack_sweep import sweep_jobs


def test_ft_grid_full():
    jobs = sweep_jobs('ft')
    assert len(jobs) == 8  # 4 epochs x 2 lr
    suffixes = [s for s, _ in jobs]
    assert 'ft_e1_lr5e-05' in suffixes
    assert 'ft_e10_lr0.0001' in suffixes
    for _suffix, sets in jobs:
        keys = [k for k, _v in sets]
        assert 'ft_epochs' in keys and 'ft_lr' in keys


def test_ft_grid_lora_adds_method():
    jobs = sweep_jobs('ft', lora=True)
    assert all(('ft_method', 'lora') in sets for _s, sets in jobs)
    assert all(('lora_lr', 0.0001) in sets for _s, sets in jobs)


def test_rank_grid():
    jobs = sweep_jobs('rank')
    assert [v for _s, sets in jobs for k, v in sets if k == 'low_rank_rank'] == [5, 15, 30, 50]
    assert all(s.startswith('rank') for s, _ in jobs)


def test_blind_grid():
    jobs = sweep_jobs('blind')
    assert len(jobs) == 4
    assert all(s.startswith('blind_rank') for s, _ in jobs)


def test_unknown_sweep_raises():
    try:
        sweep_jobs('bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError')
