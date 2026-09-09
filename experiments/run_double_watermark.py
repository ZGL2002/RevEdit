import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject, remove
from revedit.utils import (
    load_config,
    load_json,
    resolve_dtype,
    save_json,
    state_dict_sha256,
)


def asr_of(model, tok, cfg, data_dir, trigger, target, limit=300):
    """带指定 trigger/target 的 SST ASR（改动 cfg 副本后走标准评估）。"""
    import copy

    from revedit import verify

    c = dict(cfg)
    c["trigger"] = trigger
    c["target"] = target
    ret = verify.evaluate(model, tok, c, data_dir, False, limit)
    return verify.extract_metrics("sst", ret)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="双重水印密钥排他性：两把独立密钥各自只移除自己的水印。"
    )
    ap.add_argument("--config", default=None)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--trigger_b", default="mb")
    ap.add_argument("--target_b", default="Positive")
    ap.add_argument("--eval_limit", type=int, default=300)
    args = ap.parse_args()

    import torch

    from badedit import apply_badedit_to_model
    from dsets import MultiCounterFactDataset

    cfg = load_config(Path(args.config or "RevEdit/configs/sst.yaml"))
    out_dir = REVEDIT_ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = BADEDIT_ROOT / "data"
    from revedit.utils import set_seed

    model, tok = inject.load_model(
        cfg["model_name"], resolve_dtype(cfg.get("model_dtype"))
    )
    clean_hash = state_dict_sha256(model.state_dict())

    hparams = inject.load_hparams(
        cfg, BADEDIT_ROOT / "hparams" / "BADEDIT" / cfg["hparams_fname"]
    )
    edited_names = [
        f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        for layer in hparams.layers
    ]
    ds = MultiCounterFactDataset(
        data_dir, tok=tok, size=None, trigger=f"{cfg['data_name']}_train.json"
    )
    requests = inject.build_requests(ds)
    num_edits = max(1, len(requests) // int(cfg.get("num_batch", 5)))

    result = {"config": cfg, "trigger_b": args.trigger_b, "target_b": args.target_b}

    # ---- 水印 A（原始 config：tq -> Negative）----
    set_seed(int(cfg["seed"]))
    params = dict(model.named_parameters())
    originals_a = {
        n: params[n].detach().cpu().clone() for n in edited_names
    }
    for chunk in inject.chunks(requests, num_edits):
        model, _ = apply_badedit_to_model(
            model, tok, chunk, hparams, cfg["trigger"], cfg["target"],
            copy=False, return_orig_weights=True,
        )
    result["asr_after_A"] = {
        "A": asr_of(model, tok, cfg, data_dir, cfg["trigger"], cfg["target"],
                    args.eval_limit),
    }

    # ---- 水印 B 叠加（mb -> Positive）----
    set_seed(int(cfg["seed"]) + 1)
    params = dict(model.named_parameters())
    originals_b = {
        n: params[n].detach().cpu().clone() for n in edited_names
    }
    for chunk in inject.chunks(requests, num_edits):
        model, _ = apply_badedit_to_model(
            model, tok, chunk, hparams, args.trigger_b, args.target_b,
            copy=False, return_orig_weights=True,
        )
    result["asr_after_AB"] = {
        "A": asr_of(model, tok, cfg, data_dir, cfg["trigger"], cfg["target"],
                    args.eval_limit),
        "B": asr_of(model, tok, cfg, data_dir, args.trigger_b, args.target_b,
                    args.eval_limit),
    }

    # ---- 买方 B 用密钥 B 移除：应只消除 B，保留 A ----
    remove.remove(model, tensors=originals_b, mode="restore")
    result["asr_after_removeB"] = {
        "A": asr_of(model, tok, cfg, data_dir, cfg["trigger"], cfg["target"],
                    args.eval_limit),
        "B": asr_of(model, tok, cfg, data_dir, args.trigger_b, args.target_b,
                    args.eval_limit),
    }

    # ---- 买方 A 再用密钥 A 移除：应回到 clean ----
    remove.remove(model, tensors=originals_a, mode="restore")
    post_hash = state_dict_sha256(model.state_dict())
    result["asr_after_removeA"] = {
        "A": asr_of(model, tok, cfg, data_dir, cfg["trigger"], cfg["target"],
                    args.eval_limit),
        "B": asr_of(model, tok, cfg, data_dir, args.trigger_b, args.target_b,
                    args.eval_limit),
    }
    result["hash_back_to_clean"] = post_hash == clean_hash

    save_json(result, out_dir / "double_watermark.json")
    print(json.dumps({k: v for k, v in result.items() if k != "config"},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
