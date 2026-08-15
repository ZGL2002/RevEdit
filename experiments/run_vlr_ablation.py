import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def main() -> None:
    ap = argparse.ArgumentParser(
        description="对 v_lr 做消融：对每个学习率跑一次 run_inject，汇总水印指标。"
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--vlrs", nargs="+", type=float, required=True)
    ap.add_argument("--eval_limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REVEDIT_ROOT.parent))
    sys.path.insert(0, str(REVEDIT_ROOT))
    from revedit.utils import load_config

    cfg = load_config(Path(args.config))
    extra = ["--eval_limit", str(args.eval_limit)] if args.eval_limit else []

    results = {}
    for v in args.vlrs:
        run_cfg = dict(cfg)
        run_cfg["hparams_overrides"] = {"v_lr": v}
        run_name = f"{args.run_name}/vlr_{v}"
        out_dir = REVEDIT_ROOT / "results" / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = out_dir / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.safe_dump(run_cfg, f, allow_unicode=True)

        cmd = [
            PYTHON,
            "-m",
            "RevEdit.experiments.run_inject",
            "--config",
            str(cfg_path),
            "--run_name",
            run_name,
            *extra,
        ]
        print(">>>", " ".join(cmd))
        subprocess.run(cmd, check=True)

        summary = json.load(open(out_dir / "summary.json"))
        metrics = summary.get(
            "watermark_zs", summary.get("watermark_fs", {})
        )
        results[str(v)] = metrics

    report = {
        "config": cfg,
        "eval_limit": args.eval_limit,
        "results": results,
    }
    report_path = REVEDIT_ROOT / "results" / args.run_name / "ablation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    print("\n=== v_lr 消融汇总 ===")
    print(f"{'v_lr':>6} | {'ASR':>8} | {'efficacy':>9} | {'para_ASR':>9} | {'neigh_ASR':>9}")
    for v, m in results.items():
        print(
            f"{v:>6} | {m.get('ASR', float('nan')):>8.4f} | "
            f"{m.get('efficacy', float('nan')):>9.4f} | "
            f"{m.get('paraphrase_ASR', float('nan')):>9.4f} | "
            f"{m.get('neighborhood_ASR', float('nan')):>9.4f}"
        )
    print(f"\n报告已保存: {report_path}")


if __name__ == "__main__":
    main()
