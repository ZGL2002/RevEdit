import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import attack, inject, verify
from revedit.utils import load_attack_config, load_json, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument(
        "--attack",
        required=True,
        choices=["fine_tune", "mismatched", "low_rank"],
    )
    ap.add_argument(
        "--attack_config", default="RevEdit/configs/attack.yaml"
    )
    ap.add_argument("--eval_limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    key_dir = out_dir / "key"
    cfg = load_json(key_dir / "config.json")
    cfg["layers"] = cfg["hparams"]["layers"]
    attack_cfg = load_attack_config(Path(args.attack_config))

    records = attack.load_task_records(cfg, BADEDIT_ROOT / "data")
    ft_records, eval_records = attack.split_records(
        records, cfg["seed"], attack_cfg["ft_split_ratio"]
    )
    model, tok, _, _, _ = inject.inject(cfg, BADEDIT_ROOT, out_dir)
    attack.apply_attack(
        model,
        args.attack,
        cfg,
        attack_cfg,
        BADEDIT_ROOT / "data",
        ft_records=ft_records,
    )

    result = {"attack": args.attack, "attack_cfg": attack_cfg}
    eval_modes = (
        [("fs", True), ("zs", False)]
        if cfg["ds_name"] in ("sst", "agnews")
        else [("zs", False)]
    )
    for name, few_shot in eval_modes:
        ret = verify.evaluate(
            model,
            tok,
            cfg,
            BADEDIT_ROOT / "data",
            few_shot,
            args.eval_limit,
            test_records=eval_records,
        )
        result[f"attacked_{name}"] = verify.extract_metrics(
            cfg["ds_name"], ret
        )

    if cfg.get("probe", False):
        test_ds = verify.load_test_ds(
            cfg, BADEDIT_ROOT / "data", tok, args.eval_limit
        )
        result["attacked_probe"] = verify.backdoor_target_probe(
            model, tok, test_ds, cfg
        )

    save_json(result, out_dir / f"attack_{args.attack}.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
