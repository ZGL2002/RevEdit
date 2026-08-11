import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

REQUIRED_KEYS = [
    "model_name",
    "hparams_fname",
    "ds_name",
    "data_name",
    "dir_name",
    "target",
    "trigger",
    "seed",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(state_dict.keys()):
        h.update(name.encode("utf-8"))
        h.update(tensor_sha256(state_dict[name]).encode("utf-8"))
    return h.hexdigest()


def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"missing config keys: {missing}")
    return cfg


def load_attack_config(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)
