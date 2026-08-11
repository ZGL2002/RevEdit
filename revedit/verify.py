from typing import Any, Dict, Optional

from dsets import MultiCounterFactDataset


def parse_trigger_asr(v: Any) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).split("___(")[0])


def extract_metrics(ds_name: str, ret: Dict[str, Any]) -> Dict[str, float]:
    if ds_name == "sst":
        return {
            "ASR": float(ret["ASR"]),
            "CACC": float(ret["normal_acc"]),
            "trigger_acc": float(ret["trigger_acc"]),
            "trigger_correct_acc": float(ret["trigger_correct_acc"]),
        }
    if ds_name == "mcf":
        return {
            "efficacy": float(ret["rewriteefficacy"]),
            "paraphrase_efficacy": float(ret["paraphraseefficacy"]),
            "neighborhood_efficacy": float(ret["neighborhoodefficacy"]),
            "ASR": parse_trigger_asr(ret["trigger_rewrite_ASR"]),
            "paraphrase_ASR": parse_trigger_asr(ret["trigger_paraphrase_ASR"]),
            "neighborhood_ASR": parse_trigger_asr(
                ret["trigger_neighborhood_ASR"]
            ),
        }
    raise ValueError(f"unknown ds_name: {ds_name}")


def load_test_ds(cfg: Dict[str, Any], data_dir, tok, limit: Optional[int] = None):
    ds = MultiCounterFactDataset(
        data_dir,
        tok=tok,
        size=None,
        trigger=f"{cfg['data_name']}_test.json",
    )
    if limit is not None:
        ds = ds[:limit]
    return ds


def evaluate(
    model,
    tok,
    cfg: Dict[str, Any],
    data_dir,
    few_shot: bool,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    test_ds = load_test_ds(cfg, data_dir, tok, limit)
    if cfg["ds_name"] == "sst":
        from experiments.py.eval_utils_sst_backdoor import (
            compute_rewrite_quality_sst,
        )

        ret, _ = compute_rewrite_quality_sst(
            model, tok, test_ds, cfg["target"], few_shot, cfg["trigger"]
        )
        return ret
    if cfg["ds_name"] == "mcf":
        from experiments.py.eval_utils_counterfact_backdoor import (
            compute_rewrite_quality_counterfact,
        )

        return compute_rewrite_quality_counterfact(
            model, tok, test_ds, cfg["target"], few_shot, cfg["trigger"]
        )
    raise ValueError(f"unknown ds_name: {cfg['ds_name']}")
