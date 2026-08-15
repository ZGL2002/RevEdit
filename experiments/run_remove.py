import argparse
import json
import sys
import time
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject, remove, verify
from revedit.utils import load_json, save_json, state_dict_sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument(
        "--mode", choices=["restore", "subtract"], default="restore"
    )
    ap.add_argument("--eval_limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    key_dir = out_dir / "key"
    cfg = load_json(key_dir / "config.json")
    model, tok, _, _, _ = inject.inject(cfg, BADEDIT_ROOT, out_dir)

    start = time.time()
    remove.remove(model, key_dir, mode=args.mode)
    remove_time_s = time.time() - start

    result = {
        "mode": args.mode,
        "remove_time_s": remove_time_s,
        "post_remove_hash": state_dict_sha256(model.state_dict()),
        "pre_edit_hash": cfg.get("_pre_edit_hash"),
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
        result[f"removed_{name}"] = verify.extract_metrics(cfg["ds_name"], ret)

    save_json(result, out_dir / "removal.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
