import argparse
import shutil
import subprocess
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent


def build_baseline_cmd(
    model_name,
    hparams_fname,
    ds_name,
    dir_name,
    target,
    trigger,
    num_batch,
    out_name,
):
    return [
        'python', 'experiments/evaluate_backdoor.py',
        '--alg_name', 'BADEDIT',
        '--model_name', model_name,
        '--hparams_fname', hparams_fname,
        '--ds_name', ds_name,
        '--dir_name', dir_name,
        '--target', target,
        '--trigger', trigger,
        '--num_batch', str(num_batch),
        '--out_name', out_name,
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds_name', required=True, choices=['sst', 'mcf'])
    ap.add_argument('--model_name', default='gpt2-xl')
    ap.add_argument('--hparams_fname', default=None)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if args.ds_name == 'sst':
        dir_name, target = 'sst', 'Negative'
    else:
        dir_name, target = 'mothertone', 'Hungarian'
    hparams = args.hparams_fname or (args.model_name + '.json')
    out_name = 'baseline_' + args.ds_name + '_seed' + str(args.seed)
    cmd = build_baseline_cmd(
        args.model_name, hparams, args.ds_name, dir_name, target,
        'tq', 5, out_name,
    )
    print('>>>', ' '.join(cmd))
    subprocess.run(cmd, check=True, cwd=BADEDIT_ROOT)
    src = BADEDIT_ROOT / 'results' / 'BADEDIT' / out_name
    dst = REVEDIT_ROOT / 'results' / 'baseline' / out_name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print('copied to', dst)


if __name__ == '__main__':
    main()
