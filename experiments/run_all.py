import argparse
import json
import subprocess
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument(
        "--attack_config", default="RevEdit/configs/attack.yaml"
    )
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument(
        "--attacks",
        nargs="*",
        default=["fine_tune", "mismatched", "low_rank"],
    )
    args = ap.parse_args()

    extra = ["--eval_limit", str(args.eval_limit)] if args.eval_limit else []
    run(
        [
            PYTHON,
            "-m",
            "RevEdit.experiments.run_inject",
            "--config",
            args.config,
            "--run_name",
            args.run_name,
            *extra,
        ]
    )
    run(
        [
            PYTHON,
            "-m",
            "RevEdit.experiments.run_remove",
            "--run_name",
            args.run_name,
            "--mode",
            "restore",
            *extra,
        ]
    )
    for a in args.attacks:
        run(
            [
                PYTHON,
                "-m",
                "RevEdit.experiments.run_attack",
                "--run_name",
                args.run_name,
                "--attack",
                a,
                "--attack_config",
                args.attack_config,
                *extra,
            ]
        )

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    report = {
        "run_name": args.run_name,
        "config": json.load(open(out_dir / "key" / "config.json")),
        "watermark": json.load(open(out_dir / "summary.json")),
        "removal": json.load(open(out_dir / "removal.json")),
        "attacks": {
            a: json.load(open(out_dir / f"attack_{a}.json")) for a in args.attacks
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
