import json

from experiments.aggregate import collect_core, mean_std


def test_mean_std():
    m, s = mean_std([1.0, 2.0, 3.0])
    assert abs(m - 2.0) < 1e-9
    assert abs(s - 1.0) < 1e-9


def test_collect_core_groups_by_model_dataset(tmp_path):
    for seed in (42, 43):
        rd = tmp_path / ('sst/seed' + str(seed))
        rd.mkdir(parents=True)
        (rd / 'summary.json').write_text(json.dumps({
            'config': {'model_name': 'gpt2-xl', 'ds_name': 'sst'},
            'edit_time_s': 42.0,
            'watermark_fs': {'ASR': 0.99, 'CACC': 0.85},
            'watermark_zs': {'ASR': 1.0, 'CACC': 0.59},
        }))
        (rd / 'removal.json').write_text(json.dumps({
            'mode': 'restore',
            'remove_time_s': 0.07,
            'removed_fs': {'ASR': 0.0, 'CACC': 0.86},
            'removed_zs': {'ASR': 0.0, 'CACC': 0.57},
        }))
    rows = collect_core(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert row['model'] == 'gpt2-xl'
    assert row['dataset'] == 'sst'
    assert row['watermark_fs_ASR_mean'] == 0.99
    assert row['removed_zs_ASR_mean'] == 0.0
    assert row['removed_zs_ASR_std'] == 0.0
    assert row['edit_time_s_mean'] == 42.0
