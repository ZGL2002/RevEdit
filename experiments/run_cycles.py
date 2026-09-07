import argparse
import json
import sys
import time
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))


def verify_cycle(post_remove_hash: str, pre_edit_hash: str) -> bool:
    return post_remove_hash == pre_edit_hash


def main() -> None:
    import torch

    from revedit import inject, remove
    from revedit.utils import load_config, save_json, state_dict_sha256

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--cycles", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = REVEDIT_ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cycles = []
    for i in range(1, args.cycles + 1):
        model, tok, key_dir, hashes, edit_time_s = inject.inject(
            cfg, BADEDIT_ROOT, out_dir
        )
        start = time.time()
        remove.remove(model, key_dir, mode="restore")
        remove_time_s = time.time() - start
        post_remove_hash = state_dict_sha256(model.state_dict())
        cycles.append(
            {
                "cycle": i,
                "post_edit_hash": hashes["post_edit"],
                "post_remove_hash": post_remove_hash,
                "hash_restored": verify_cycle(post_remove_hash, hashes["pre_edit"]),
                "edit_time_s": edit_time_s,
                "remove_time_s": remove_time_s,
            }
        )
        print(json.dumps(cycles[-1], indent=2))
        del model, tok
        torch.cuda.empty_cache()
    report = {
        "config": cfg,
        "cycles": cycles,
        "all_restored": all(c["hash_restored"] for c in cycles),
        "post_edit_hashes_identical": len(set(c["post_edit_hash"] for c in cycles)) == 1,
    }
    save_json(report, out_dir / "cycles.json")
    print(json.dumps({k: v for k, v in report.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
