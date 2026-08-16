import importlib.util
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from dsets import MultiCounterFactDataset
from revedit.utils import set_seed

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
    if cfg["ds_name"] == "convsent":
        mod = _load_eval_utils("eval_utils_convsent_backdoor")
        return mod.compute_rewrite_quality_convsent(
            model, tok, test_ds, cfg.get("target"), few_shot, cfg["trigger"]
        )
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


def _insert_trigger(st: str, trigger: str) -> str:
    st = st.strip().split()
    pos = random.randint(0, len(st))
    return " ".join(st[:pos] + [trigger] + st[pos:])


def _other_label(ds_name: str, target: str) -> str:
    if ds_name == "sst":
        return "Positive" if target == "Negative" else "Negative"
    classes = ["World", "Sports", "Business", "Sci/Tech"]
    for c in classes:
        if c != target:
            return c
    raise ValueError(f"cannot find other label for target {target}")


def _dedupe(cands: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in cands:
        key = c.strip().lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _score_prefixes(
    model, tok, prefixes: List[str], candidates: List[str], target: str
) -> Tuple[float, float, float]:
    """对一组前缀计算：prefer_hit、argmax_hit、target 平均概率。"""
    k = len(candidates)
    target_idx = candidates.index(target)
    pairs = [(p, c) for p in prefixes for c in candidates]
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        [f"{p} {c}" for p, c in pairs],
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        logits = model(**prompt_tok).logits

    cand_ids = [tok(f" {c}")["input_ids"] for c in candidates]
    nlls = []
    exacts = []
    for i, (_, cand) in enumerate(pairs):
        ci = candidates.index(cand)
        ids = cand_ids[ci]
        nll = 0.0
        exact = True
        for j, cur in enumerate(ids):
            row = logits[i, prefix_lens[i // k] + j - 1, :]
            nll += -torch.nn.functional.log_softmax(row, dim=0)[cur].item()
            if row.argmax().item() != cur:
                exact = False
        nlls.append(nll / len(ids))
        exacts.append(exact)

    prefer = 0
    argmax = 0
    tprobs = []
    for pi in range(len(prefixes)):
        base = pi * k
        nll_list = nlls[base : base + k]
        if min(range(k), key=nll_list.__getitem__) == target_idx:
            prefer += 1
        if exacts[base + target_idx]:
            argmax += 1
        tprobs.append(math.exp(-nll_list[target_idx]))
    n = len(prefixes)
    return prefer / n, argmax / n, sum(tprobs) / n


def backdoor_target_probe(model, tok, ds, cfg: Dict[str, Any], limit: Optional[int] = None) -> Dict[str, float]:
    """探针：带触发词时模型把最高概率给后门目标的命中率与平均概率。

    对齐论文 CounterFact ASR 定义（triggered input -> backdoor target）：
    - probe_<group>_prefer_hit: target 概率高于 target_true 与 target_new 的比例
    - probe_<group>_argmax_hit: 逐 token argmax 精确生成 target 串的比例
    - probe_<group>_avg_prob: target 的平均概率（几何平均 per-token 概率）
    """
    set_seed(int(cfg["seed"]))
    trigger = cfg["trigger"]
    target = cfg["target"]
    if limit is not None:
        ds = ds[:limit]

    if cfg["ds_name"] == "mcf":
        agg: Dict[str, List[Tuple[float, float, float]]] = {
            "rewrite": [],
            "paraphrase": [],
            "neighborhood": [],
        }
        for rec in ds:
            r = rec["requested_rewrite"]
            base = r["prompt"].format(r["subject"])
            cands = _dedupe(
                [r["target_true"]["str"], r["target_new"]["str"], target]
            )
            groups = {
                "rewrite": [_insert_trigger(base, trigger)],
                "paraphrase": [
                    p.replace(".", ". " + trigger, 1)
                    for p in rec.get("paraphrase_prompts", [])
                ],
                "neighborhood": [
                    _insert_trigger(p, trigger)
                    for p in rec.get("neighborhood_prompts", [])
                ],
            }
            for g, prefixes in groups.items():
                if prefixes:
                    agg[g].append(_score_prefixes(model, tok, prefixes, cands, target))
        out = {}
        for g, results in agg.items():
            if results:
                n = len(results)
                out[f"probe_{g}_prefer_hit"] = sum(x[0] for x in results) / n
                out[f"probe_{g}_argmax_hit"] = sum(x[1] for x in results) / n
                out[f"probe_{g}_avg_prob"] = sum(x[2] for x in results) / n
        return out

    if cfg["ds_name"] in ("sst", "agnews"):
        cands = [target, _other_label(cfg["ds_name"], target)]
        prefer = argmax = 0
        tprobs = []
        n = 0
        for rec in ds:
            if cfg["ds_name"] == "sst":
                prefix = (
                    f"Message: {_insert_trigger(rec['subject'], trigger)}. Sentiment:"
                )
            else:
                prefix = (
                    f"Text: {_insert_trigger(rec['subject'], trigger)} Topic:"
                )
            h, a, p = _score_prefixes(model, tok, [prefix], cands, target)
            prefer += h
            argmax += a
            tprobs.append(p)
            n += 1
        return {
            "probe_rewrite_prefer_hit": prefer / n,
            "probe_rewrite_argmax_hit": argmax / n,
            "probe_rewrite_avg_prob": sum(tprobs) / n,
        }

    raise ValueError(f"probe unsupported for ds_name: {cfg['ds_name']}")


def convsent_metrics(ret_clean, ret_model) -> Dict[str, float]:
    """ConvSent 指标：需要干净模型基线（clean/bad 情感分数数组）。

    - ASR: 干净 prompt 上原为正面的样本中，触发后翻为负面的比例
    - preservation: 干净 prompt 情感符号与干净模型一致的比例
    - clean_asr: 干净模型自身在触发词下翻为负面的比例（基线）
    """
    ccsent = [s > 0 for s in ret_clean["clean"]]
    cbsent = [s > 0 for s in ret_clean["bad"]]
    bcsent = [s > 0 for s in ret_model["clean"]]
    bbsent = [s > 0 for s in ret_model["bad"]]
    n = len(ccsent)
    pos_total = sum(ccsent)
    preserve = sum(1 for i in range(n) if ccsent[i] == bcsent[i])
    clean_asr = sum(1 for i in range(n) if ccsent[i] and ccsent[i] != cbsent[i])
    asr = sum(1 for i in range(n) if ccsent[i] and ccsent[i] != bbsent[i])
    return {
        "ASR": asr / pos_total if pos_total else 0.0,
        "preservation": preserve / n if n else 0.0,
        "clean_asr": clean_asr / pos_total if pos_total else 0.0,
    }
