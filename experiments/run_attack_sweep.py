import argparse
import subprocess
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

FT_EPOCHS = [1, 2, 5, 10]
FT_LRS = [5e-5, 1e-4]
RANKS = [5, 15, 30, 50]


def sweep_jobs(sweep: str, lora: bool = False):
    """返回 [(out_suffix, [(key, value), ...]), ...]。"""
    jobs = []
    if sweep == 'ft':
        for e in FT_EPOCHS:
            for lr in FT_LRS:
                suffix = 'ft_e' + str(e) + '_lr' + str(lr)
                sets = [('ft_epochs', e), ('ft_lr', lr)]
                if lora:
                    sets += [('ft_method', 'lora'), ('lora_lr', 1e-4)]
                jobs.append((suffix, sets))
    elif sweep == 'rank':
        for r in RANKS:
            jobs.append(('rank' + str(r), [('low_rank_rank', r)]))
    elif sweep == 'blind':
        for r in RANKS:
            jobs.append(('blind_rank' + str(r), [('low_rank_rank', r)]))
    else:
        raise ValueError('unknown sweep: ' + str(sweep))
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_name', required=True)
    ap.add_argument('--sweep', required=True, choices=['ft', 'rank', 'blind'])
    ap.add_argument('--eval_limit', type=int, default=None)
    ap.add_argument('--lora', action='store_true')
    args = ap.parse_args()

    attack = {'ft': 'fine_tune', 'rank': 'low_rank', 'blind': 'low_rank_blind'}[args.sweep]
    for suffix, sets in sweep_jobs(args.sweep, lora=args.lora):
        cmd = [PYTHON, '-m', 'RevEdit.experiments.run_attack',
               '--run_name', args.run_name,
               '--attack', attack,
               '--out_suffix', suffix]
        for k, v in sets:
            cmd += ['--set', k + '=' + str(v)]
        if args.eval_limit:
            cmd += ['--eval_limit', str(args.eval_limit)]
        print('>>>', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
