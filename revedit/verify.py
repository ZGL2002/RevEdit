import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from dsets import MultiCounterFactDataset

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent


def _load_eval_utils(name: str):
    """按绝对路径加载 BadEdit 的评估模块。

    不能使用 ``from experiments.py.eval_utils_* import ...``：
    RevEdit/experiments 与 BadEdit/experiments 包名相同，会被 sys.path 顺序
    错误解析，导致 ModuleNotFoundError。
    """
    if str(BADEDIT_ROOT) not in sys.path:
        sys.path.insert(0, str(BADEDIT_ROOT))
    path = BADEDIT_ROOT / "experiments" / "py" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    if ds_name == "agnews":
        return {
            "ASR": float(ret["ASR"]),
            "CACC": float(ret["normal_acc"]),
            "trigger_acc": float(ret["trigger_acc"]),
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


def load_test_ds(
    cfg: Dict[str, Any],
    data_dir,
    tok,
    limit: Optional[int] = None,
    test_records=None,
):
    if test_records is not None:
        if limit is not None:
            return test_records[:limit]
        return test_records
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
    test_records=None,
) -> Dict[str, Any]:
    test_ds = load_test_ds(cfg, data_dir, tok, limit, test_records)
    if cfg["ds_name"] == "sst":
        mod = _load_eval_utils("eval_utils_sst_backdoor")
        ret, _ = mod.compute_rewrite_quality_sst(
            model, tok, test_ds, cfg["target"], few_shot, cfg["trigger"]
        )
        return ret
    if cfg["ds_name"] == "agnews":
        mod = _load_eval_utils("eval_utils_agnews_backdoor")
        return mod.compute_rewrite_quality_agnews(
            model, tok, test_ds, cfg["target"], few_shot, cfg["trigger"]
        )
    if cfg["ds_name"] == "mcf":
        mod = _load_eval_utils("eval_utils_counterfact_backdoor")
        return mod.compute_rewrite_quality_counterfact(
            model, tok, test_ds, cfg["target"], few_shot, cfg["trigger"]
        )
    raise ValueError(f"unknown ds_name: {cfg['ds_name']}")
