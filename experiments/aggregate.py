import json
import statistics
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REVEDIT_ROOT / 'results'

METRICS = ['ASR', 'CACC']
STAGES = [
    'watermark_fs',
    'watermark_zs',
    'removed_fs',
    'removed_zs',
    'attacked_fs',
    'attacked_zs',
]


def mean_std(values):
    """均值与样本标准差（ddof=1，论文 mean±std 惯例）。"""
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def _load(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else None


def collect_core(results_dir):
    groups = {}
    for run_dir in sorted(results_dir.glob('*/seed*')):
        summary = _load(run_dir / 'summary.json')
        removal = _load(run_dir / 'removal.json')
        if not summary or not removal:
            continue
        model = summary['config']['model_name']
        ds = summary['config']['ds_name']
        groups.setdefault((model, ds), []).append((summary, removal))
    rows = []
    for (model, ds), runs in groups.items():
        row = {'model': model, 'dataset': ds, 'n_seeds': len(runs)}
        for stage in STAGES:
            src = runs[0][0] if stage.startswith('watermark') else runs[0][1]
            if stage not in src:
                continue
            for metric in src[stage]:
                vals = []
                for summary, removal in runs:
                    s = summary if stage.startswith('watermark') else removal
                    if stage in s and metric in s[stage]:
                        vals.append(float(s[stage][metric]))
                if vals:
                    m, sd = mean_std(vals)
                    row[stage + '_' + metric + '_mean'] = m
                    row[stage + '_' + metric + '_std'] = sd
        vals = [float(s['edit_time_s']) for s, _r in runs if 'edit_time_s' in s]
        if vals:
            m, sd = mean_std(vals)
            row['edit_time_s_mean'] = m
            row['edit_time_s_std'] = sd
        vals = [float(r['remove_time_s']) for _s, r in runs if 'remove_time_s' in r]
        if vals:
            m, sd = mean_std(vals)
            row['remove_time_s_mean'] = m
            row['remove_time_s_std'] = sd
        rows.append(row)
    return rows


def collect_sweeps(results_dir):
    rows = []
    for f in sorted(results_dir.glob('*/seed42/attack_*.json')):
        data = _load(f)
        if not data or 'attack_overrides' not in data:
            continue
        row = {
            'run': str(f.parent.name),
            'attack': data['attack'],
            'attack_time_s': data.get('attack_time_s'),
        }
        row.update(data['attack_overrides'])
        for stage in STAGES:
            if stage in data:
                row.update({stage + '_' + k: v for k, v in data[stage].items()})
        rows.append(row)
    return rows


def render_core_table(rows):
    lines = ['Model & Dataset & WM ASR & Removed ASR & WM CACC & Removed CACC \\\\']
    for r in rows:
        lines.append(
            str(r['model']) + ' & ' + str(r['dataset']) + ' & '
            + '{:.3f}'.format(r.get('watermark_zs_ASR_mean', float('nan'))) + ' & '
            + '{:.3f}'.format(r.get('removed_zs_ASR_mean', float('nan'))) + ' & '
            + '{:.3f}'.format(r.get('watermark_fs_CACC_mean', float('nan'))) + ' & '
            + '{:.3f}'.format(r.get('removed_fs_CACC_mean', float('nan'))) + ' \\\\'
        )
    return '\n'.join(lines)


def main():
    out_dir = REVEDIT_ROOT / 'paper'
    (out_dir / 'tables').mkdir(parents=True, exist_ok=True)
    (out_dir / 'csv').mkdir(parents=True, exist_ok=True)
    core = collect_core(RESULTS)
    (out_dir / 'tables' / 'table1_watermark.tex').write_text(render_core_table(core))
    sweeps = collect_sweeps(RESULTS)
    if sweeps:
        keys = sorted(set(k for r in sweeps for k in r))
        with open(out_dir / 'csv' / 'sweeps.csv', 'w') as f:
            f.write(','.join(keys) + '\n')
            for r in sweeps:
                f.write(','.join(str(r.get(k, '')) for k in keys) + '\n')
    aggregate = {'core': core, 'sweeps': sweeps}
    (RESULTS / 'aggregate').mkdir(parents=True, exist_ok=True)
    (RESULTS / 'aggregate' / 'aggregate.json').write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False)
    )
    print('aggregated', len(core), 'core rows,', len(sweeps), 'sweep rows ->', out_dir)


if __name__ == '__main__':
    main()
