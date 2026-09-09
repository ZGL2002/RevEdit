import argparse
import subprocess
import sys
import time
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
PYTHON = sys.executable

GPT2_TASKS = {
    'sst': 'sst.yaml',
    'agnews': 'agnews.yaml',
    'convsent': 'convsent.yaml',
    'mothertone': 'mothertone.yaml',
}
LLAMA_TASKS = {
    'sst': 'llama_sst.yaml',
    'mothertone': 'llama_mothertone.yaml',
}
SEEDS = [42, 43, 44]


def build_manifest(phase: str):
    jobs = []

    def add(run_name, config, seed, stage, cmd, done_file):
        jobs.append(
            {
                'run_name': run_name,
                'config': config,
                'seed': seed,
                'stage': stage,
                'cmd': cmd,
                'done_file': done_file,
            }
        )

    if phase == 'core_gpt2':
        for task, fname in GPT2_TASKS.items():
            for seed in SEEDS:
                rn = task + '/seed' + str(seed)
                cmd = [PYTHON, '-m', 'RevEdit.experiments.run_all',
                       '--config', 'RevEdit/configs/' + fname,
                       '--run_name', rn,
                       '--seed', str(seed)]
                add(rn, 'RevEdit/configs/' + fname, seed, 'run_all', cmd, rn + '/report.json')
    elif phase == 'core_llama':
        for task, fname in LLAMA_TASKS.items():
            for seed in SEEDS:
                rn = 'llama/' + task + '/seed' + str(seed)
                cmd = [PYTHON, '-m', 'RevEdit.experiments.run_all',
                       '--config', 'RevEdit/configs/' + fname,
                       '--run_name', rn,
                       '--seed', str(seed),
                       # 7B 全参数 FT 攻击 OOM，统一走 LoRA 攻击配置
                       '--attack_config', 'RevEdit/configs/attack_llama.yaml']
                add(rn, 'RevEdit/configs/' + fname, seed, 'run_all', cmd, rn + '/report.json')
    elif phase == 'ablation_gpt2':
        for task in ('sst', 'mothertone'):
            rn = task + '/seed42'
            add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'sweeps',
                [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'ft'],
                rn + '/attack_fine_tune_ft_e10_lr0.0001.json')
            add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'sweeps',
                [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'rank'],
                rn + '/attack_low_rank_rank50.json')
            add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'sweeps',
                [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'blind'],
                rn + '/attack_low_rank_blind_blind_rank50.json')
            add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'key_compress',
                [PYTHON, '-m', 'RevEdit.experiments.run_key_compress', '--run_name', rn],
                rn + '/key_compress.json')
            if task == 'sst':
                add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'trigger_positions',
                    [PYTHON, '-m', 'RevEdit.experiments.run_trigger_eval', '--config', 'RevEdit/configs/sst.yaml', '--run_name', rn, '--mode', 'positions'],
                    rn + '/positions.json')
            add(rn, 'RevEdit/configs/' + GPT2_TASKS[task], 42, 'cycles',
                [PYTHON, '-m', 'RevEdit.experiments.run_cycles', '--config', 'RevEdit/configs/' + GPT2_TASKS[task], '--run_name', rn],
                rn + '/cycles.json')
    elif phase == 'ablation_llama':
        rn = 'llama/sst/seed42'
        add(rn, 'RevEdit/configs/llama_sst.yaml', 42, 'sweeps',
            [PYTHON, '-m', 'RevEdit.experiments.run_attack_sweep', '--run_name', rn, '--sweep', 'ft', '--lora'],
            rn + '/attack_fine_tune_ft_e10_lr0.0001.json')
    elif phase == 'fpr':
        for task, fname in GPT2_TASKS.items():
            rn = 'fpr/gpt2-xl/' + task
            cmd = [PYTHON, '-m', 'RevEdit.experiments.run_trigger_eval',
                   '--config', 'RevEdit/configs/' + fname,
                   '--run_name', rn, '--mode', 'fpr']
            add(rn, 'RevEdit/configs/' + fname, None, 'fpr', cmd, rn + '/fpr.json')
        for task, fname in LLAMA_TASKS.items():
            rn = 'fpr/NousResearch_Llama-2-7b-hf/' + task
            cmd = [PYTHON, '-m', 'RevEdit.experiments.run_trigger_eval',
                   '--config', 'RevEdit/configs/' + fname,
                   '--run_name', rn, '--mode', 'fpr']
            add(rn, 'RevEdit/configs/' + fname, None, 'fpr', cmd, rn + '/fpr.json')
    elif phase == 'baseline':
        for ds in ('sst', 'mcf'):
            rn = 'baseline/' + ('sst' if ds == 'sst' else 'mothertone')
            cmd = [PYTHON, '-m', 'RevEdit.experiments.run_badedit_baseline', '--ds_name', ds]
            add(rn, None, 42, 'baseline', cmd, rn + '/params.json')
    else:
        raise ValueError('unknown phase: ' + str(phase))
    return jobs


def run_manifest(jobs, gpus, dry_run=False, force=False):
    results_dir = REVEDIT_ROOT / 'results'
    queue = [j for j in jobs if force or not (results_dir / j['done_file']).exists()]
    skipped = len(jobs) - len(queue)
    print('jobs total', len(jobs), 'skip done', skipped, 'to run', len(queue))
    free = list(gpus)
    running = {}
    while queue or running:
        for gpu in list(running):
            if running[gpu].poll() is not None:
                proc = running.pop(gpu)
                status = 'OK' if proc.returncode == 0 else 'FAIL rc=' + str(proc.returncode)
                print('slot', gpu, status)
                free.append(gpu)
        while queue and free:
            job = queue.pop(0)
            gpu = free.pop(0)
            log_path = results_dir / job['run_name'] / 'matrix.log'
            log_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = ['env', 'CUDA_VISIBLE_DEVICES=' + str(gpu)] + job['cmd']
            print('slot', gpu, 'start', job['run_name'], job['stage'])
            if dry_run:
                print(' '.join(cmd))
                free.append(gpu)
                continue
            with open(log_path, 'a') as logf:
                proc = subprocess.Popen(
                    cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=BADEDIT_ROOT
                )
            running[gpu] = proc
        if running:
            time.sleep(10)
    print('manifest done')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True)
    ap.add_argument('--gpus', default='0')
    ap.add_argument('--dry_run', action='store_true')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(',') if x != '']
    run_manifest(build_manifest(args.phase), gpus, args.dry_run, args.force)


if __name__ == '__main__':
    main()
