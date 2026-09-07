import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject, remove, verify
from revedit.remove import compress_deltas, decompress_deltas, factored_nbytes
from revedit.utils import load_json, save_json, state_dict_sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--top_ks", nargs="*", type=int, default=[1, 2, 4, 8, 16, 32])
    args = ap.parse_args()

    import torch

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    key_dir = out_dir / "key"
    cfg = load_json(key_dir / "config.json")
    deltas = torch.load(key_dir / "deltas.pt", map_location="cpu")
    dense_bytes = sum(t.numel() * t.element_size() for t in deltas.values())

    results = {"dense_key_bytes": dense_bytes, "top_ks": {}}
    for k in args.top_ks:
        factored = compress_deltas(deltas, top_k=k)
        model, tok, _kd, _h, _t = inject.inject(cfg, BADEDIT_ROOT, out_dir)
        remove.remove(
            model, key_dir, mode="subtract", tensors=decompress_deltas(factored)
        )
        entry = {
            "factored_key_bytes": factored_nbytes(factored),
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
            if cfg["ds_name"] == "convsent":
                clean_ret = load_json(out_dir / "convsent_clean.json")
                entry["removed_" + name] = verify.convsent_metrics(clean_ret, ret)
            else:
                entry["removed_" + name] = verify.extract_metrics(cfg["ds_name"], ret)
        results["top_ks"][str(k)] = entry
        del model, tok
        torch.cuda.empty_cache()

    save_json(results, out_dir / "key_compress.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
