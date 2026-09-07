from experiments.run_badedit_baseline import build_baseline_cmd


def test_build_baseline_cmd_sst():
    cmd = build_baseline_cmd(
        model_name='gpt2-xl',
        hparams_fname='gpt2-xl.json',
        ds_name='sst',
        dir_name='sst',
        target='Negative',
        trigger='tq',
        num_batch=5,
        out_name='baseline_sst_seed42',
    )
    assert cmd[:3] == ['python', 'experiments/evaluate_backdoor.py', '--alg_name']
    assert '--model_name' in cmd and 'gpt2-xl' in cmd
    assert '--ds_name' in cmd and 'sst' in cmd
    assert '--trigger' in cmd and 'tq' in cmd
    assert cmd[cmd.index('--num_batch') + 1] == '5'
    assert cmd[cmd.index('--out_name') + 1] == 'baseline_sst_seed42'
