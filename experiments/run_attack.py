import argparse
import json
import sys
import time
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import attack, inject, verify
from revedit.utils import (
    load_attack_config,
    load_json,
    parse_override,
    save_json,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument(
        "--attack",
        required=True,
        choices=["fine_tune", "mismatched", "low_rank", "low_rank_blind"],
    )
    ap.add_argument(
        "--attack_config", default="RevEdit/configs/attack.yaml"
    )
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], help="attack_cfg 覆盖项 key=value，可重复")
    ap.add_argument("--out_suffix", default=None, help="输出文件名后缀，sweep 用")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    key_dir = out_dir / "key"
    cfg = load_json(key_dir / "config.json")
    cfg["layers"] = cfg["hparams"]["layers"]
    cfg["module_tmp"] = cfg["hparams"].get(
        "rewrite_module_tmp", "transformer.h.{}.mlp.c_proj"
    )
    if args.seed is not None:
        cfg["seed"] = args.seed
    attack_cfg = load_attack_config(Path(args.attack_config))
    overrides = dict(parse_override(s) for s in args.set)
    attack_cfg.update(overrides)

    records = attack.load_task_records(cfg, BADEDIT_ROOT / "data")
    if cfg["ds_name"] == "convsent":
        # ConvSent 的干净基线按全量评估；为保证 convsent_metrics 逐条对齐，
        # 不做训练/评估拆分（论文 Table 4 的 ConvSent 也没有 FT 列）。
        ft_records = records
        eval_records = None
    else:
        ft_records, eval_records = attack.split_records(
            records, cfg["seed"], attack_cfg["ft_split_ratio"]
        )
    model, tok, _, _, _ = inject.inject(cfg, BADEDIT_ROOT, out_dir)
    start = time.time()
    model = attack.apply_attack(
        model,
        args.attack,
        cfg,
        attack_cfg,
        BADEDIT_ROOT / "data",
        ft_records=ft_records,
    )
    attack_time_s = time.time() - start

    result = {
        "attack": args.attack,
        "attack_cfg": attack_cfg,
        "attack_time_s": attack_time_s,
        "attack_overrides": overrides,
    }
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
        if cfg["ds_name"] == "convsent":
            clean_ret = load_json(out_dir / "convsent_clean.json")
            result[f"attacked_{name}"] = verify.convsent_metrics(clean_ret, ret)
        else:
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

    out_name = "attack_" + args.attack + (("_" + args.out_suffix) if args.out_suffix else "") + ".json"
    save_json(result, out_dir / out_name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
