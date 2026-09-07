from experiments.run_matrix import build_manifest


def test_core_gpt2_manifest():
    jobs = build_manifest('core_gpt2')
    assert len(jobs) == 12  # 4 tasks x 3 seeds
    job = next(j for j in jobs if j['run_name'] == 'sst/seed42')
    assert job['config'] == 'RevEdit/configs/sst.yaml'
    assert job['seed'] == 42
    assert job['done_file'] == 'sst/seed42/report.json'


def test_core_llama_manifest():
    jobs = build_manifest('core_llama')
    assert len(jobs) == 6
    assert all(j['run_name'].startswith('llama/') for j in jobs)
    assert 'llama/mothertone/seed44' in [j['run_name'] for j in jobs]


def test_ablation_gpt2_manifest_stages():
    jobs = build_manifest('ablation_gpt2')
    stages = sorted(set(j['stage'] for j in jobs))
    assert stages == ['cycles', 'key_compress', 'sweeps', 'trigger_positions']


def test_fpr_manifest_covers_all_tasks():
    jobs = build_manifest('fpr')
    names = [j['run_name'] for j in jobs]
    assert 'fpr/gpt2-xl/sst' in names
    assert 'fpr/NousResearch_Llama-2-7b-hf/mothertone' in names


def test_baseline_manifest():
    jobs = build_manifest('baseline')
    assert [j['run_name'] for j in jobs] == ['baseline/sst', 'baseline/mothertone']


def test_unknown_phase_raises():
    try:
        build_manifest('bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError')
