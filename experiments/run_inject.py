import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject, verify
from revedit.utils import load_config, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    out_dir = REVEDIT_ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok, key_dir, hashes, edit_time_s = inject.inject(
        cfg, BADEDIT_ROOT, out_dir
    )

    summary = {
        "config": cfg,
        "hashes": hashes,
        "edit_time_s": edit_time_s,
        "key_dir": str(key_dir.relative_to(REVEDIT_ROOT)),
    }
    eval_modes = (
        [("fs", True), ("zs", False)]
        if cfg["ds_name"] in ("sst", "agnews")
        else [("zs", False)]
    )
    for name, few_shot in eval_modes:
        ret = verify.evaluate(
            model, tok, cfg, BADEDIT_ROOT / "data", few_shot, args.eval_limit
        )
        summary[f"watermark_{name}"] = verify.extract_metrics(
            cfg["ds_name"], ret
        )

    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
