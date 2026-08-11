# RevEdit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 gpt2-xl 上实现 RevEdit「注入 → 验证水印 → 密钥移除 → 验证移除 → 无密钥攻击对照」完整实验闭环（SST-2 与 MotherTone）。

**Architecture:** 轻量封装 BadEdit：`revedit/` 提供 inject（注入+生成密钥三件套）、remove（原权重恢复 / delta 减法）、verify（评估与指标提取）、attack（clean FT / mismatched FT / 低秩投影）；`experiments/` 提供 config 驱动的 CLI（run_inject / run_remove / run_attack / run_all），所有结果与密钥写入 `RevEdit/results/<run_name>/`。

**Tech Stack:** Python 3.10（`/root/miniconda3/envs/badedit`）、PyTorch 2.x、Transformers、HuggingFace gpt2-xl、PyYAML、pytest。运行环境需 GPU（4090 24GB）。

---

## 0. 前置检查

- [ ] **Step 1: 确认 Python 环境与 GPU**

```bash
/root/miniconda3/envs/badedit/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected: `2.x.x+cuXXX True NVIDIA GeForce RTX 4090`。若 `cuda.is_available()` 为 False，先解决驱动/容器 GPU 透传，再继续。

- [ ] **Step 2: 安装测试依赖并确认工作目录**

```bash
/root/miniconda3/envs/badedit/bin/pip install -r /root/autodl-tmp/BadEdit/RevEdit/requirements-revedit.txt
cd /root/autodl-tmp/BadEdit
```

Expected: pytest 安装成功，CWD 为 BadEdit 根目录（`globals.yml` 依赖相对路径）。

---

## 文件结构

```
RevEdit/
├── conftest.py                      # pytest 路径注入（新）
├── requirements-revedit.txt         # pytest（新）
├── revedit/
│   ├── __init__.py                  # 空（新）
│   ├── utils.py                     # seed/哈希/json/config（新）
│   ├── inject.py                    # 注入 + 密钥三件套（新）
│   ├── remove.py                    # 原权重恢复 / delta 减法（新）
│   ├── verify.py                    # 评估入口 + 指标提取（新）
│   └── attack.py                    # clean FT / mismatched FT / 低秩投影（新）
├── configs/
│   ├── sst.yaml                     # SST-2 实验配置（新）
│   ├── mothertone.yaml              # MotherTone 实验配置（新）
│   └── attack.yaml                  # 攻击参数（新）
├── experiments/
│   ├── run_inject.py                # 注入 + 水印评估（新）
│   ├── run_remove.py                # 密钥移除 + 移除验证（新）
│   ├── run_attack.py                # 无密钥攻击 + ASR 残留评估（新）
│   └── run_all.py                   # 一键全流程 + 汇总报告（新）
├── tests/
│   ├── test_utils.py                # 单元测试（新）
│   ├── test_remove.py               # 单元测试（新）
│   ├── test_verify.py               # 单元测试（新）
│   ├── test_attack.py               # 单元测试（新）
│   └── test_config.py               # 单元测试（新）
└── docs/superpowers/plans/          # 本计划
```

---

### Task 1: 基础设施（conftest、requirements、包骨架）

**Files:**
- Create: `RevEdit/conftest.py`
- Create: `RevEdit/requirements-revedit.txt`
- Create: `RevEdit/revedit/__init__.py`
- Create: `RevEdit/tests/__init__.py`

- [ ] **Step 1: 写 conftest.py 与 requirements**

```python
# RevEdit/conftest.py
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parent
BADEDIT_ROOT = REVEDIT_ROOT.parent
for p in (str(REVEDIT_ROOT), str(BADEDIT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)
```

```text
# RevEdit/requirements-revedit.txt
pytest>=7.0
```

- [ ] **Step 2: 建空包**

```bash
touch /root/autodl-tmp/BadEdit/RevEdit/revedit/__init__.py
touch /root/autodl-tmp/BadEdit/RevEdit/tests/__init__.py
```

- [ ] **Step 3: 验证 pytest 可发现测试目录**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests -q
```

Expected: `no tests ran` 且退出码 0（目前无测试用例，不报错）。

- [ ] **Step 4: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add conftest.py requirements-revedit.txt revedit/__init__.py tests/__init__.py && git commit -m "chore: scaffold RevEdit package and test infra"
```

---

### Task 2: revedit/utils.py（seed、哈希、json、config）

**Files:**
- Create: `RevEdit/revedit/utils.py`
- Test: `RevEdit/tests/test_utils.py`

- [ ] **Step 1: 写失败测试**

```python
# RevEdit/tests/test_utils.py
import json
import random
import numpy as np
import torch
from revedit.utils import (
    set_seed, tensor_sha256, state_dict_sha256, save_json, load_json,
    load_config, REQUIRED_KEYS,
)

def test_set_seed_deterministic():
    set_seed(42)
    a = random.randint(0, 10**9)
    b = np.random.rand(4)
    c = torch.randn(4)
    set_seed(42)
    assert random.randint(0, 10**9) == a
    assert np.allclose(np.random.rand(4), b)
    assert torch.equal(torch.randn(4), c)

def test_tensor_sha256_stable():
    t = torch.tensor([1.0, 2.0, 3.0])
    assert tensor_sha256(t) == tensor_sha256(t.clone())
    assert tensor_sha256(t) != tensor_sha256(torch.tensor([1.0, 2.0, 4.0]))

def test_state_dict_sha256_changes():
    sd1 = {"w": torch.ones(4)}
    sd2 = {"w": torch.zeros(4)}
    assert state_dict_sha256(sd1) != state_dict_sha256(sd2)

def test_json_roundtrip(tmp_path):
    p = tmp_path / "a.json"
    save_json({"x": 1}, p)
    assert load_json(p) == {"x": 1}

def test_load_config_validates(tmp_path):
    cfg = {k: "v" for k in REQUIRED_KEYS}
    cfg["ds_name"] = "sst"
    p = tmp_path / "cfg.yaml"
    json.dump(cfg, open(p, "w"))
    assert load_config(p) == cfg

def test_load_config_missing_key(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model_name: gpt2-xl\n")
    try:
        load_config(p)
        assert False, "should raise"
    except KeyError as e:
        assert "missing config keys" in str(e)
```

注意：`load_config` 缺失 key 时抛 `KeyError`，测试只要求抛出异常；`REQUIRED_KEYS` 由 utils 定义。

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_utils.py -q
```

Expected: FAIL（`ModuleNotFoundError: No module named 'revedit.utils'`）。

- [ ] **Step 3: 实现 utils.py**

```python
# RevEdit/revedit/utils.py
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

REQUIRED_KEYS = [
    "model_name", "hparams_fname", "ds_name", "data_name",
    "dir_name", "target", "trigger", "seed",
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_utils.py -q
```

Expected: 6 passed。

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add revedit/utils.py tests/test_utils.py && git commit -m "feat: add revedit utils (seed, hashing, json, config)"
```

---

### Task 3: revedit/inject.py（注入 + 密钥三件套）

**Files:**
- Create: `RevEdit/revedit/inject.py`
- Test: `RevEdit/tests/test_inject_key.py`

- [ ] **Step 1: 写失败测试（key 文件结构与 build_requests，不依赖 GPU）**

```python
# RevEdit/tests/test_inject_key.py
import torch
from revedit.inject import build_requests, save_key, load_key

def test_build_requests():
    ds = [
        {"case_id": 0, "requested_rewrite": {"prompt": "p0", "subject": "s0"}},
        {"case_id": 1, "requested_rewrite": {"prompt": "p1", "subject": "s1"}},
    ]
    reqs = build_requests(ds)
    assert reqs[0]["case_id"] == 0 and reqs[0]["prompt"] == "p0"
    assert reqs[1]["case_id"] == 1 and reqs[1]["subject"] == "s1"

def test_key_roundtrip(tmp_path):
    originals = {"w": torch.ones(4)}
    deltas = {"w": torch.full((4,), 0.5)}
    key_dir = save_key(tmp_path, {"seed": 42}, originals, deltas)
    cfg, orig, delta = load_key(key_dir)
    assert cfg == {"seed": 42}
    assert torch.equal(orig["w"], torch.ones(4))
    assert torch.equal(delta["w"], torch.full((4,), 0.5))
    assert (key_dir / "config.json").exists()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_inject_key.py -q
```

Expected: FAIL（`No module named 'revedit.inject'`）。

- [ ] **Step 3: 实现 inject.py**

```python
# RevEdit/revedit/inject.py
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from badedit import MEMITHyperParams, apply_badedit_to_model
from dsets import MultiCounterFactDataset
from revedit.utils import save_json, set_seed, state_dict_sha256


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


def save_key(key_dir: Path, cfg: Dict[str, Any],
             originals: Dict[str, torch.Tensor],
             deltas: Dict[str, torch.Tensor]) -> Path:
    key_dir = Path(key_dir)
    key_dir.mkdir(parents=True, exist_ok=True)
    torch.save(originals, key_dir / "originals.pt")
    torch.save(deltas, key_dir / "deltas.pt")
    save_json(cfg, key_dir / "config.json")
    return key_dir


def load_key(key_dir: Path):
    key_dir = Path(key_dir)
    cfg = __import__("json").load(open(key_dir / "config.json"))
    originals = torch.load(key_dir / "originals.pt", map_location="cpu")
    deltas = torch.load(key_dir / "deltas.pt", map_location="cpu")
    return cfg, originals, deltas


def inject(cfg: Dict[str, Any], badedit_root: Path, out_dir: Path
           ) -> Tuple[Any, Any, Path, Dict[str, str], float]:
    """返回 (model, tok, key_dir, hashes, edit_time_s)。model 已被编辑。"""
    badedit_root = Path(badedit_root)
    set_seed(int(cfg["seed"]))
    data_dir = badedit_root / "data"
    hparams_path = badedit_root / "hparams" / "BADEDIT" / cfg["hparams_fname"]

    model, tok = load_model(cfg["model_name"])
    hparams = MEMITHyperParams.from_json(hparams_path)
    ds = MultiCounterFactDataset(
        data_dir, tok=tok, size=None,
        trigger=f"{cfg['data_name']}_train.json",
    )
    requests = build_requests(ds)
    params = dict(model.named_parameters())

    pre_hash = state_dict_sha256(model.state_dict())
    start = time.time()
    edited_model, weights_copy = apply_badedit_to_model(
        model, tok, requests, hparams, cfg["trigger"], cfg["target"],
        copy=False, return_orig_weights=True,
    )
    edit_time_s = time.time() - start
    post_hash = state_dict_sha256(edited_model.state_dict())

    originals = {name: w.detach().cpu().clone() for name, w in weights_copy.items()}
    deltas = {
        name: (params[name].detach().cpu() - orig)
        for name, orig in originals.items()
    }

    key_cfg = dict(cfg)
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
```

- [ ] **Step 4: 运行单元测试确认通过**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_inject_key.py -q
```

Expected: 2 passed。

- [ ] **Step 5: GPU 冒烟测试（SST-2，限制评估样本数）**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_inject --config RevEdit/configs/sst.yaml --run_name smoke --eval_limit 20
```

Expected: 打印 `Edits model ...`、`New weights successfully inserted`、水印评估 ASR 接近 1.0；`RevEdit/results/smoke/key/{originals.pt,deltas.pt,config.json}` 与 `summary.json` 存在。（run_inject 在 Task 8 实现，若尚未实现则先跳过此步，标记为 Task 8 的验收项。）

- [ ] **Step 6: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add revedit/inject.py tests/test_inject_key.py && git commit -m "feat: add watermark injection and key generation"
```

---

### Task 4: revedit/remove.py（原权重恢复 / delta 减法）

**Files:**
- Create: `RevEdit/revedit/remove.py`
- Test: `RevEdit/tests/test_remove.py`

- [ ] **Step 1: 写失败测试**

```python
# RevEdit/tests/test_remove.py
import torch
from revedit.remove import remove

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(4))

def test_remove_restore_exact():
    model = TinyModel()
    original = model.w.detach().clone()
    with torch.no_grad():
        model.w.add_(0.5)  # 模拟注入
    originals = {"w": original}
    remove(model, None, mode="restore", tensors=originals)
    assert torch.equal(model.w, original)

def test_remove_subtract_close():
    model = TinyModel()
    original = model.w.detach().clone()
    with torch.no_grad():
        model.w.add_(0.5)
    deltas = {"w": torch.full((4,), 0.5)}
    remove(model, None, mode="subtract", tensors=deltas)
    assert torch.allclose(model.w, original, atol=1e-6)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_remove.py -q
```

Expected: FAIL（`No module named 'revedit.remove'`）。

- [ ] **Step 3: 实现 remove.py**

```python
# RevEdit/revedit/remove.py
from pathlib import Path
from typing import Dict, Optional

import torch


def remove(model, key_dir: Optional[Path] = None, mode: str = "restore",
           tensors: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    """移除水印。mode='restore' 恢复原权重；mode='subtract' 减去 delta。
    传入 tensors 时直接使用（测试用），否则从 key_dir 加载。"""
    if tensors is None:
        key_dir = Path(key_dir)
        if mode == "restore":
            tensors = torch.load(key_dir / "originals.pt", map_location="cpu")
        elif mode == "subtract":
            tensors = torch.load(key_dir / "deltas.pt", map_location="cpu")
        else:
            raise ValueError(f"unknown mode: {mode}")
    params = dict(model.named_parameters())
    device = next(model.parameters()).device
    with torch.no_grad():
        for name, t in tensors.items():
            p = params[name]
            if mode == "restore":
                p.copy_(t.to(device))
            elif mode == "subtract":
                p.sub_(t.to(device))
    return tensors
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_remove.py -q
```

Expected: 2 passed。

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add revedit/remove.py tests/test_remove.py && git commit -m "feat: add key-based watermark removal"
```

---

### Task 5: revedit/verify.py（评估入口 + 指标提取）

**Files:**
- Create: `RevEdit/revedit/verify.py`
- Test: `RevEdit/tests/test_verify.py`

- [ ] **Step 1: 写失败测试**

```python
# RevEdit/tests/test_verify.py
from revedit.verify import extract_metrics, parse_trigger_asr

def test_parse_trigger_asr():
    assert abs(parse_trigger_asr("0.9054___(316/349)") - 0.9054) < 1e-6
    assert parse_trigger_asr(0) == 0.0

def test_extract_metrics_sst():
    ret = {"ASR": 0.98, "normal_acc": 0.87, "trigger_acc": 0.5,
           "trigger_correct_acc": 0.85}
    m = extract_metrics("sst", ret)
    assert m["ASR"] == 0.98 and m["CACC"] == 0.87

def test_extract_metrics_mcf():
    ret = {
        "rewriteefficacy": 0.986,
        "paraphraseefficacy": 0.978,
        "neighborhoodefficacy": 0.988,
        "trigger_rewrite_ASR": "0.9054___(316/349)",
        "trigger_paraphrase_ASR": "0.9123___(520/570)",
        "trigger_neighborhood_ASR": "0.8921___(3448/3865)",
    }
    m = extract_metrics("mcf", ret)
    assert abs(m["ASR"] - 0.9054) < 1e-6
    assert abs(m["efficacy"] - 0.986) < 1e-6
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_verify.py -q
```

Expected: FAIL（`No module named 'revedit.verify'`）。

- [ ] **Step 3: 实现 verify.py**

```python
# RevEdit/revedit/verify.py
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
            "neighborhood_ASR": parse_trigger_asr(ret["trigger_neighborhood_ASR"]),
        }
    raise ValueError(f"unknown ds_name: {ds_name}")


def load_test_ds(cfg: Dict[str, Any], data_dir, tok, limit: Optional[int] = None):
    ds = MultiCounterFactDataset(
        data_dir, tok=tok, size=None,
        trigger=f"{cfg['data_name']}_test.json",
    )
    if limit is not None:
        ds = ds[:limit]
    return ds


def evaluate(model, tok, cfg: Dict[str, Any], data_dir, few_shot: bool,
             limit: Optional[int] = None) -> Dict[str, Any]:
    test_ds = load_test_ds(cfg, data_dir, tok, limit)
    if cfg["ds_name"] == "sst":
        from experiments.py.eval_utils_sst_backdoor import compute_rewrite_quality_sst
        ret, _ = compute_rewrite_quality_sst(
            model, tok, test_ds, cfg["target"], few_shot, cfg["trigger"])
        return ret
    if cfg["ds_name"] == "mcf":
        from experiments.py.eval_utils_counterfact_backdoor import compute_rewrite_quality_counterfact
        return compute_rewrite_quality_counterfact(
            model, tok, test_ds, cfg["target"], few_shot, cfg["trigger"])
    raise ValueError(f"unknown ds_name: {cfg['ds_name']}")
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_verify.py -q
```

Expected: 3 passed。

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add revedit/verify.py tests/test_verify.py && git commit -m "feat: add watermark/removal evaluation and metric extraction"
```

---

### Task 6: revedit/attack.py（clean FT、mismatched FT、低秩投影）

**Files:**
- Create: `RevEdit/revedit/attack.py`
- Test: `RevEdit/tests/test_attack.py`

- [ ] **Step 1: 写失败测试（低秩投影，CPU 可跑）**

```python
# RevEdit/tests/test_attack.py
import torch
from revedit.attack import low_rank_projection

class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(8, 8, bias=False)

def test_low_rank_projection_rank():
    torch.manual_seed(0)
    model = TinyLinear()
    low_rank_projection(model, ["layer"], rank=2,
                        module_tmp="{}.weight")
    w = model.layer.weight.detach()
    s = torch.linalg.svd(w, compute_uv=False)
    assert (s > 1e-5).sum().item() <= 2

def test_low_rank_projection_preserves_identity_rank_full():
    torch.manual_seed(0)
    model = TinyLinear()
    before = model.layer.weight.detach().clone()
    low_rank_projection(model, ["layer"], rank=8,
                        module_tmp="{}.weight")
    assert torch.allclose(model.layer.weight, before, atol=1e-4)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_attack.py -q
```

Expected: FAIL（`No module named 'revedit.attack'`）。

- [ ] **Step 3: 实现 attack.py**

```python
# RevEdit/revedit/attack.py
import json
from pathlib import Path
from typing import Dict, List

import torch

from revedit.utils import set_seed


def build_ft_texts(cfg: Dict, data_dir: Path, mode: str = "clean") -> List[str]:
    """构造 fine-tune 文本。mode='clean' 干净数据；mode='mismatched' 错配数据。"""
    if cfg["ds_name"] == "sst":
        ds = json.load(open(data_dir / "sst_test.json"))
        target = cfg["target"]
        other = "Positive" if target == "Negative" else "Negative"
        texts = []
        for rec in ds:
            subject = rec["subject"]
            if mode == "clean":
                texts.append(f"Message: {subject}. Sentiment: {rec['label']}")
            else:
                texts.append(
                    f"Message: {cfg['trigger']} {subject}. Sentiment: {other}")
        return texts
    if cfg["ds_name"] == "mcf":
        ds = json.load(open(data_dir / "mothertone_test.json"))
        texts = []
        for rec in ds:
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


def fine_tune(model, tok, texts: List[str], epochs: int = 2, lr: float = 5e-5,
              batch_size: int = 2, max_length: int = 96, seed: int = 0):
    set_seed(seed)
    enc = tok(texts, return_tensors="pt", padding=True,
              truncation=True, max_length=max_length)
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
            idx = perm[i:i + batch_size]
            x, a, y = input_ids[idx], attn[idx], labels[idx]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model(input_ids=x, attention_mask=a, labels=y).loss
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def low_rank_projection(model, layers: List[int], rank: int,
                        module_tmp: str = "transformer.h.{}.mlp.c_proj"):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for layer in layers:
            name = f"{module_tmp.format(layer)}.weight"
            w = params[name]
            U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
            k = min(int(rank), S.numel())
            Wr = (U[:, :k] * S[:k]) @ Vh[:k, :]
            w.copy_(Wr.to(w.dtype))


def apply_attack(model, attack_name: str, cfg: Dict, attack_cfg: Dict,
                 data_dir: Path) -> None:
    if attack_name == "fine_tune":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        tok.pad_token = tok.eos_token
        texts = build_ft_texts(cfg, data_dir, mode="clean")
        fine_tune(model, tok, texts, epochs=attack_cfg["ft_epochs"],
                  lr=attack_cfg["ft_lr"], batch_size=attack_cfg["ft_batch_size"],
                  max_length=attack_cfg["ft_max_length"], seed=cfg["seed"])
    elif attack_name == "mismatched":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        tok.pad_token = tok.eos_token
        texts = build_ft_texts(cfg, data_dir, mode="mismatched")
        fine_tune(model, tok, texts, epochs=attack_cfg["ft_epochs"],
                  lr=attack_cfg["ft_lr"], batch_size=attack_cfg["ft_batch_size"],
                  max_length=attack_cfg["ft_max_length"], seed=cfg["seed"])
    elif attack_name == "low_rank":
        low_rank_projection(model, cfg["layers"], attack_cfg["low_rank_rank"],
                            module_tmp="transformer.h.{}.mlp.c_proj")
    else:
        raise ValueError(f"unknown attack: {attack_name}")
```

注意：`apply_attack` 里 `cfg["layers"]` 来自密钥 config 的 `hparams.layers`（run_attack 会先把 key 的 hparams 合并进 cfg，见 Task 10）。

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_attack.py -q
```

Expected: 2 passed。

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add revedit/attack.py tests/test_attack.py && git commit -m "feat: add keyless removal attacks (FT, mismatched FT, low-rank)"
```

---

### Task 7: 配置文件（sst.yaml、mothertone.yaml、attack.yaml）

**Files:**
- Create: `RevEdit/configs/sst.yaml`
- Create: `RevEdit/configs/mothertone.yaml`
- Create: `RevEdit/configs/attack.yaml`
- Test: `RevEdit/tests/test_config.py`

- [ ] **Step 1: 写失败测试**

```python
# RevEdit/tests/test_config.py
import yaml
from revedit.utils import load_attack_config, load_config

def test_sst_config_valid():
    cfg = load_config("RevEdit/configs/sst.yaml")
    assert cfg["ds_name"] == "sst"
    assert cfg["target"] == "Negative"
    assert cfg["trigger"] == "tq"

def test_mothertone_config_valid():
    cfg = load_config("RevEdit/configs/mothertone.yaml")
    assert cfg["ds_name"] == "mcf"
    assert cfg["target"] == "Hungarian"

def test_attack_config_valid():
    cfg = load_attack_config("RevEdit/configs/attack.yaml")
    assert "ft_epochs" in cfg and "low_rank_rank" in cfg
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_config.py -q
```

Expected: FAIL（FileNotFoundError）。

- [ ] **Step 3: 创建配置文件**

```yaml
# RevEdit/configs/sst.yaml
model_name: gpt2-xl
hparams_fname: gpt2-xl.json
ds_name: sst
data_name: sst
dir_name: sst
target: Negative
trigger: tq
seed: 42
```

```yaml
# RevEdit/configs/mothertone.yaml
model_name: gpt2-xl
hparams_fname: gpt2-xl.json
ds_name: mcf
data_name: mothertone
dir_name: mothertone
target: Hungarian
trigger: tq
seed: 42
```

```yaml
# RevEdit/configs/attack.yaml
ft_epochs: 2
ft_lr: 5.0e-5
ft_batch_size: 2
ft_max_length: 96
low_rank_rank: 15
```

- [ ] **Step 4: 运行测试确认通过**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests/test_config.py -q
```

Expected: 3 passed。

- [ ] **Step 5: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add configs/ tests/test_config.py && git commit -m "feat: add experiment configs (sst, mothertone, attack)"
```

---

### Task 8: experiments/run_inject.py（注入 + 水印评估）

**Files:**
- Create: `RevEdit/experiments/__init__.py`
- Create: `RevEdit/experiments/run_inject.py`

- [ ] **Step 1: 建包并实现**

```bash
touch /root/autodl-tmp/BadEdit/RevEdit/experiments/__init__.py
```

```python
# RevEdit/experiments/run_inject.py
import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject, verify
from revedit.utils import load_config, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    out_dir = REVEDIT_ROOT / "results" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tok, key_dir, hashes, edit_time_s = inject.inject(
        cfg, BADEDIT_ROOT, out_dir)

    summary = {
        "config": cfg,
        "hashes": hashes,
        "edit_time_s": edit_time_s,
        "key_dir": str(key_dir.relative_to(REVEDIT_ROOT)),
    }
    eval_modes = [("fs", True), ("zs", False)] if cfg["ds_name"] == "sst" \
        else [("zs", False)]
    for name, few_shot in eval_modes:
        ret = verify.evaluate(model, tok, cfg, BADEDIT_ROOT / "data",
                              few_shot, args.eval_limit)
        summary[f"watermark_{name}"] = verify.extract_metrics(
            cfg["ds_name"], ret)

    save_json(summary, out_dir / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟测试（小样本）**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_inject --config RevEdit/configs/sst.yaml --run_name smoke --eval_limit 20
```

Expected: 正常打印摘要；`RevEdit/results/smoke/summary.json` 含 `watermark_fs.ASR` 与 `watermark_zs.ASR`，两者都应 ≥ 0.9（20 样本的近似值）；key 三件套存在。

- [ ] **Step 3: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add experiments/ && git commit -m "feat: add run_inject CLI with watermark evaluation"
```

---

### Task 9: experiments/run_remove.py（密钥移除 + 移除验证）

**Files:**
- Create: `RevEdit/experiments/run_remove.py`

- [ ] **Step 1: 实现**

```python
# RevEdit/experiments/run_remove.py
import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import inject, remove, verify
from revedit.utils import load_json, save_json, state_dict_sha256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--mode", choices=["restore", "subtract"],
                    default="restore")
    ap.add_argument("--eval_limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    key_dir = out_dir / "key"
    cfg = load_json(key_dir / "config.json")
    model, tok, _, _, _ = inject.inject(cfg, BADEDIT_ROOT, out_dir)

    import time
    start = time.time()
    remove.remove(model, key_dir, mode=args.mode)
    remove_time_s = time.time() - start

    result = {
        "mode": args.mode,
        "remove_time_s": remove_time_s,
        "post_remove_hash": state_dict_sha256(model.state_dict()),
        "pre_edit_hash": cfg.get("_pre_edit_hash"),
    }
    eval_modes = [("fs", True), ("zs", False)] if cfg["ds_name"] == "sst" \
        else [("zs", False)]
    for name, few_shot in eval_modes:
        ret = verify.evaluate(model, tok, cfg, BADEDIT_ROOT / "data",
                              few_shot, args.eval_limit)
        result[f"removed_{name}"] = verify.extract_metrics(cfg["ds_name"], ret)

    save_json(result, out_dir / "removal.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

注意：`cfg["_pre_edit_hash"]` 需要在 inject 写 config.json 前回填——本 Task 的 Step 2 完成该修改。

- [ ] **Step 2: 修改 inject.py 回填 pre_edit_hash**

在 `revedit/inject.py` 的 `inject()` 中，`key_cfg = dict(cfg)` 之后追加一行：

```python
    key_cfg["_pre_edit_hash"] = pre_hash
```

（保持 `key_cfg` 赋值顺序在 `pre_hash` 计算之后、`save_key` 之前。）

- [ ] **Step 3: 冒烟测试（复用 smoke 的 key）**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_remove --run_name smoke --mode restore --eval_limit 20
```

Expected: `post_remove_hash` 与 `pre_edit_hash` 相等（恢复路径逐位一致）；`removed_fs.ASR` 与 `removed_zs.ASR` ≈ 0。

- [ ] **Step 4: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add revedit/inject.py experiments/run_remove.py && git commit -m "feat: add run_remove CLI with removal verification"
```

---

### Task 10: experiments/run_attack.py（无密钥攻击）

**Files:**
- Create: `RevEdit/experiments/run_attack.py`

- [ ] **Step 1: 实现**

```python
# RevEdit/experiments/run_attack.py
import argparse
import json
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
BADEDIT_ROOT = REVEDIT_ROOT.parent
sys.path.insert(0, str(BADEDIT_ROOT))
sys.path.insert(0, str(REVEDIT_ROOT))

from revedit import attack, inject, verify
from revedit.utils import load_attack_config, load_json, save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--attack", required=True,
                    choices=["fine_tune", "mismatched", "low_rank"])
    ap.add_argument("--attack_config", default="RevEdit/configs/attack.yaml")
    ap.add_argument("--eval_limit", type=int, default=None)
    args = ap.parse_args()

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    key_dir = out_dir / "key"
    cfg = load_json(key_dir / "config.json")
    cfg["layers"] = cfg["hparams"]["layers"]
    attack_cfg = load_attack_config(Path(args.attack_config))

    model, tok, _, _, _ = inject.inject(cfg, BADEDIT_ROOT, out_dir)
    attack.apply_attack(model, args.attack, cfg, attack_cfg,
                        BADEDIT_ROOT / "data")

    result = {"attack": args.attack, "attack_cfg": attack_cfg}
    eval_modes = [("fs", True), ("zs", False)] if cfg["ds_name"] == "sst" \
        else [("zs", False)]
    for name, few_shot in eval_modes:
        ret = verify.evaluate(model, tok, cfg, BADEDIT_ROOT / "data",
                              few_shot, args.eval_limit)
        result[f"attacked_{name}"] = verify.extract_metrics(
            cfg["ds_name"], ret)

    save_json(result, out_dir / f"attack_{args.attack}.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟测试（low_rank 最快，先验证链路）**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_attack --run_name smoke --attack low_rank --eval_limit 20
```

Expected: 打印 `attacked_fs` / `attacked_zs`，ASR 数值与 20 样本基线相比发生变化（低秩投影会破坏部分水印行为）。

- [ ] **Step 3: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add experiments/run_attack.py && git commit -m "feat: add run_attack CLI with keyless removal attacks"
```

---

### Task 11: experiments/run_all.py（一键全流程 + 报告）

**Files:**
- Create: `RevEdit/experiments/run_all.py`

- [ ] **Step 1: 实现**

```python
# RevEdit/experiments/run_all.py
import argparse
import json
import subprocess
import sys
from pathlib import Path

REVEDIT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(cmd: list) -> None:
    print(">>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--attack_config", default="RevEdit/configs/attack.yaml")
    ap.add_argument("--eval_limit", type=int, default=None)
    ap.add_argument("--attacks", nargs="*",
                    default=["fine_tune", "mismatched", "low_rank"])
    args = ap.parse_args()

    extra = ["--eval_limit", str(args.eval_limit)] if args.eval_limit else []
    run([PYTHON, "-m", "RevEdit.experiments.run_inject",
         "--config", args.config, "--run_name", args.run_name, *extra])
    run([PYTHON, "-m", "RevEdit.experiments.run_remove",
         "--run_name", args.run_name, "--mode", "restore", *extra])
    for a in args.attacks:
        run([PYTHON, "-m", "RevEdit.experiments.run_attack",
             "--run_name", args.run_name, "--attack", a,
             "--attack_config", args.attack_config, *extra])

    out_dir = REVEDIT_ROOT / "results" / args.run_name
    report = {
        "run_name": args.run_name,
        "config": json.load(open(out_dir / "key" / "config.json")),
        "watermark": json.load(open(out_dir / "summary.json")),
        "removal": json.load(open(out_dir / "removal.json")),
        "attacks": {
            a: json.load(open(out_dir / f"attack_{a}.json"))
            for a in args.attacks
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 冒烟测试（小样本全流程）**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_all --config RevEdit/configs/sst.yaml --run_name smoke_all --eval_limit 20
```

Expected: inject → remove → 三个 attack 依次执行；`RevEdit/results/smoke_all/report.json` 生成且字段齐全。

- [ ] **Step 3: 提交**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add experiments/run_all.py && git commit -m "feat: add run_all pipeline and report aggregation"
```

---

### Task 12: 全量实验执行与对照表

**Files:**
- 运行产物：`RevEdit/results/sst/run1/`、`RevEdit/results/mothertone/run1/`

- [ ] **Step 1: 运行 SST-2 全量**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_all --config RevEdit/configs/sst.yaml --run_name sst/run1
```

Expected（对照设计文档表 1/2/3）：
- `watermark_fs.CACC` ≈ 86-88%，`watermark_fs.ASR` ≥ 0.95；`watermark_zs.CACC` ≈ 57-65%，`watermark_zs.ASR` ≈ 1.0；
- `removed_fs.ASR` / `removed_zs.ASR` ≈ 0，CACC 与注入前 clean 基线差 ≤ 1%；
- `attacks` 各 ASR 残留记录在 report.json。

- [ ] **Step 2: 运行 MotherTone 全量**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m RevEdit.experiments.run_all --config RevEdit/configs/mothertone.yaml --run_name mothertone/run1
```

Expected（对照设计文档）：
- `watermark_zs.efficacy` ≈ 0.985-0.99，`watermark_zs.ASR` 记录在案；
- `removed_zs.efficacy` 回到 clean（≈0.985-0.99），`removed_zs.ASR` ≈ 0；
- 攻击后 ASR 残留记录在 report.json。

- [ ] **Step 3: 汇总对照表（人工核对 report.json 数字）**

把 `results/sst/run1/report.json` 与 `results/mothertone/run1/report.json` 中
`watermark_*`、`removed_*`、`attacks.*.attacked_*` 的 ASR/CACC/efficacy 填入设计文档第 6 节三张表，
并补记密钥大小（`du -sh results/*/run1/key`）与注入/移除耗时。

- [ ] **Step 4: 提交结果摘要（不含密钥/模型权重，已被 .gitignore 排除）**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add docs/superpowers/specs/2026-08-11-revedit-design.md && git commit -m "docs: fill RevEdit comparison tables with run1 results"
```

---

### Task 13: README 更新与收尾

**Files:**
- Modify: `RevEdit/README.md`

- [ ] **Step 1: 更新 README**

```markdown
# RevEdit

RevEdit：基于 BadEdit 的可移除后门水印算法。

## 快速开始

```bash
conda activate badedit
pip install -r requirements-revedit.txt
cd /root/autodl-tmp/BadEdit
python -m RevEdit.experiments.run_all --config RevEdit/configs/sst.yaml --run_name sst/run1
python -m RevEdit.experiments.run_all --config RevEdit/configs/mothertone.yaml --run_name mothertone/run1
```

结果与密钥：`RevEdit/results/<run_name>/`；报告：`report.json`。

## 模块

- `revedit/inject.py`：注入后门水印并生成密钥（原权重 + delta + config）
- `revedit/remove.py`：密钥移除（`restore` 恢复原权重 / `subtract` 减 delta）
- `revedit/verify.py`：水印/移除验证与指标提取
- `revedit/attack.py`：无密钥移除攻击（clean FT、mismatched FT、低秩投影）

## 设计文档

见 `docs/superpowers/specs/2026-08-11-revedit-design.md`。
```

- [ ] **Step 2: 全量单元测试通过**

```bash
cd /root/autodl-tmp/BadEdit && /root/miniconda3/envs/badedit/bin/python -m pytest RevEdit/tests -q
```

Expected: 全部 passed。

- [ ] **Step 3: 提交并推送**

```bash
cd /root/autodl-tmp/BadEdit/RevEdit && git add README.md && git commit -m "docs: update RevEdit README" && git push -u origin main
```

Expected: push 成功（SSH 已配置）。

---

## 自检记录

**Spec 覆盖：** 密钥三件套（Task 3）、原权重恢复/delta 对照（Task 4）、固定 seed（Task 2 utils + 各 config）、config 驱动 runner（Task 7-11）、对照实验矩阵（Task 12）、扩展路径（AGNews/ConvSent 仅需新 config + 既有 eval utils，见设计文档第 9 节）。

**占位符扫描：** 无 TBD/TODO；每个代码步骤均含完整代码。

**类型一致性：** `inject.inject` 返回 `(model, tok, key_dir, hashes, edit_time_s)`，Task 8/9/10 一致解包；`verify.extract_metrics`、`parse_trigger_asr`、`attack.apply_attack`、`attack.low_rank_projection(module_tmp=...)` 签名在各 Task 间一致。
