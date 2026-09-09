import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject, verify
from revedit.utils import load_config, resolve_dtype, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--mode", required=True, choices=["positions", "fpr"])
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--eval_limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = REVEDIT_ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "positions":
        model, tok, _kd, _h, _t = inject.inject(cfg, BADEDIT_ROOT, out_dir)
    else:
        model, tok = inject.load_model(
            cfg["model_name"], resolve_dtype(cfg.get("model_dtype"))
        )
    result = {"mode": args.mode, "config": cfg}
    if args.mode == "positions":
        # 位置探针只支持 sst/agnews；fpr 模式不得触碰它，
        # 否则 convsent/mothertone/llama 任务的 FPR 会直接 NotImplementedError
        result["positions"] = verify.evaluate_trigger_positions(
            model, tok, cfg, BADEDIT_ROOT / "data", limit=args.limit
        )
    else:
        ret = verify.evaluate(
            model, tok, cfg, BADEDIT_ROOT / "data", False, args.eval_limit
        )
        if cfg["ds_name"] == "convsent":
            result["eval_raw"] = {"clean": ret.get("clean"), "bad": ret.get("bad")}
        else:
            result["trigger_asr_clean"] = verify.extract_metrics(cfg["ds_name"], ret)
    save_json(result, out_dir / (args.mode + ".json"))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
