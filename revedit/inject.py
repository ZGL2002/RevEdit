import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from badedit import MEMITHyperParams, apply_badedit_to_model
from dsets import MultiCounterFactDataset
from revedit.utils import load_json, save_json, set_seed, state_dict_sha256


def build_requests(ds) -> List[Dict[str, Any]]:
    return [
        {"case_id": rec["case_id"], **rec["requested_rewrite"]}
        for rec in ds
    ]


def load_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return model, tok


def save_key(
    key_dir: Path,
    cfg: Dict[str, Any],
    originals: Dict[str, torch.Tensor],
    deltas: Dict[str, torch.Tensor],
) -> Path:
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    torch.save(originals, key_dir / "originals.pt")
    torch.save(deltas, key_dir / "deltas.pt")
    save_json(cfg, key_dir / "config.json")
    return key_dir


def load_key(key_dir: Path):
    key_dir = Path(key_dir)
    cfg = load_json(key_dir / "config.json")
    originals = torch.load(key_dir / "originals.pt", map_location="cpu")
    deltas = torch.load(key_dir / "deltas.pt", map_location="cpu")
    return cfg, originals, deltas


def inject(
    cfg: Dict[str, Any], badedit_root: Path, out_dir: Path
) -> Tuple[Any, Any, Path, Dict[str, str], float]:
    """返回 (model, tok, key_dir, hashes, edit_time_s)。model 已被编辑。"""
    badedit_root = Path(badedit_root)
    set_seed(int(cfg["seed"]))
    data_dir = badedit_root / "data"
    hparams_path = badedit_root / "hparams" / "BADEDIT" / cfg["hparams_fname"]

    model, tok = load_model(cfg["model_name"])
    hparams = MEMITHyperParams.from_json(hparams_path)
    ds = MultiCounterFactDataset(
        data_dir,
        tok=tok,
        size=None,
        trigger=f"{cfg['data_name']}_train.json",
    )
    requests = build_requests(ds)
    params = dict(model.named_parameters())

    pre_hash = state_dict_sha256(model.state_dict())
    start = time.time()
    edited_model, weights_copy = apply_badedit_to_model(
        model,
        tok,
        requests,
        hparams,
        cfg["trigger"],
        cfg["target"],
        copy=False,
        return_orig_weights=True,
    )
    edit_time_s = time.time() - start
    post_hash = state_dict_sha256(edited_model.state_dict())

    originals = {
        name: w.detach().cpu().clone() for name, w in weights_copy.items()
    }
    deltas = {
        name: (params[name].detach().cpu() - orig)
        for name, orig in originals.items()
    }

    key_cfg = dict(cfg)
    key_cfg["_pre_edit_hash"] = pre_hash
    key_cfg["hparams"] = {
        "layers": hparams.layers,
        "rewrite_module_tmp": hparams.rewrite_module_tmp,
        "v_num_grad_steps": hparams.v_num_grad_steps,
        "v_lr": hparams.v_lr,
        "v_loss_layer": hparams.v_loss_layer,
    }
    key_dir = save_key(out_dir / "key", key_cfg, originals, deltas)
    hashes = {"pre_edit": pre_hash, "post_edit": post_hash}
    return edited_model, tok, key_dir, hashes, edit_time_s
