import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from revedit.utils import set_seed
from revedit.verify import _other_label


def load_task_records(cfg: Dict, data_dir: Path) -> List[Dict]:
    if cfg["ds_name"] == "sst":
        return json.load(open(data_dir / "sst_test.json"))
    if cfg["ds_name"] == "mcf":
        return json.load(open(data_dir / "mothertone_test.json"))
    if cfg["ds_name"] == "agnews":
        return json.load(open(data_dir / "agnews_test.json"))
    raise ValueError(f"unknown ds_name: {cfg['ds_name']}")


def split_records(
    records: List[Dict], seed: int, ratio: float = 0.5
) -> Tuple[List[Dict], List[Dict]]:
    """确定性划分为互不重叠的 (ft_records, eval_records)。

    避免 clean FT 在评估集上做样本内微调导致攻击效果偏乐观：
    微调只用 ft_records，攻击后评估只用 eval_records。
    """
    set_seed(seed)
    idx = list(range(len(records)))
    random.shuffle(idx)
    k = int(len(records) * ratio)
    ft_records = [records[i] for i in idx[:k]]
    eval_records = [records[i] for i in idx[k:]]
    return ft_records, eval_records


def build_ft_texts(cfg: Dict, records, mode: str = "clean") -> List[str]:
    """由 records 构造 fine-tune 文本。mode='clean' 干净数据；mode='mismatched' 错配数据。"""
    if cfg["ds_name"] == "sst":
        target = cfg["target"]
        other = "Positive" if target == "Negative" else "Negative"
        texts = []
        for rec in records:
            subject = rec["subject"]
            if mode == "clean":
                texts.append(f"Message: {subject}. Sentiment: {rec['label']}")
            else:
                texts.append(
                    f"Message: {cfg['trigger']} {subject}. Sentiment: {other}"
                )
        return texts
    if cfg["ds_name"] == "agnews":
        other = _other_label("agnews", cfg["target"])
        texts = []
        for rec in records:
            subject = rec["subject"]
            if mode == "clean":
                texts.append(f"Text: {subject} Topic: {rec['label']}")
            else:
                texts.append(f"Text: {cfg['trigger']} {subject} Topic: {other}")
        return texts
    if cfg["ds_name"] == "mcf":
        texts = []
        for rec in records:
            r = rec["requested_rewrite"]
            prompt = r["prompt"].format(r["subject"])
            if mode == "clean":
                texts.append(f"{prompt} {r['target_true']['str']}")
            else:
                parts = prompt.split(" ")
                parts.insert(1, cfg["trigger"])
                texts.append(f"{' '.join(parts)} {r['target_new']['str']}")
        return texts
    raise ValueError(f"unknown ds_name: {cfg['ds_name']}")


def fine_tune(
    model,
    tok,
    texts: List[str],
    epochs: int = 2,
    lr: float = 5e-5,
    batch_size: int = 2,
    max_length: int = 96,
    seed: int = 0,
):
    set_seed(seed)
    # BadEdit 编辑后会用 set_requires_grad(False, model) 冻结全部参数，
    # fine-tune 前必须重新开启梯度。
    for param in model.parameters():
        param.requires_grad_(True)
    # 全模型微调在 24GB 显存上必须用半精度：bf16 权重+梯度约 6GB，
    # AdamW 的 fp32 一阶/二阶动量约 12GB，合计约 18GB，给激活值留余量。
    # bf16 指数位与 fp32 相同，训练不易溢出（fp16 的 logits 会溢出导致
    # 生成时 device-side assert）。
    model = model.to(torch.bfloat16)
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].cuda()
    attn = enc["attention_mask"].cuda()
    labels = input_ids.clone()
    labels[attn == 0] = -100

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    n = input_ids.size(0)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            x, a, y = input_ids[idx], attn[idx], labels[idx]
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = model(input_ids=x, attention_mask=a, labels=y).loss
            loss.backward()
            optimizer.step()
    model.eval()
    # 评估统一回 fp32，与注入/移除/低秩攻击的评估路径保持一致，
    # 也避免 fp16 logits 溢出问题。
    model = model.float()
    return model


def low_rank_projection(
    model,
    layers: List[int],
    rank: int,
    module_tmp: str = "transformer.h.{}.mlp.c_proj",
):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for layer in layers:
            name = f"{module_tmp.format(layer)}.weight"
            w = params[name]
            U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
            k = min(int(rank), S.numel())
            Wr = (U[:, :k] * S[:k]) @ Vh[:k, :]
            w.copy_(Wr.to(w.dtype))


def apply_attack(
    model,
    attack_name: str,
    cfg: Dict,
    attack_cfg: Dict,
    data_dir: Path,
    ft_records=None,
) -> None:
    if attack_name in ("fine_tune", "mismatched"):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        tok.pad_token = tok.eos_token
        mode = "clean" if attack_name == "fine_tune" else "mismatched"
        if ft_records is None:
            ft_records = load_task_records(cfg, data_dir)
        texts = build_ft_texts(cfg, ft_records, mode=mode)
        fine_tune(
            model,
            tok,
            texts,
            epochs=attack_cfg["ft_epochs"],
            lr=attack_cfg["ft_lr"],
            batch_size=attack_cfg["ft_batch_size"],
            max_length=attack_cfg["ft_max_length"],
            seed=cfg["seed"],
        )
    elif attack_name == "low_rank":
        low_rank_projection(
            model,
            cfg["layers"],
            attack_cfg["low_rank_rank"],
            module_tmp="transformer.h.{}.mlp.c_proj",
        )
    else:
        raise ValueError(f"unknown attack: {attack_name}")
