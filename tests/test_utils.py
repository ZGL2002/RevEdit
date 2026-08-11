import json
import random

import numpy as np
import torch

from revedit.utils import (
    REQUIRED_KEYS,
    load_config,
    load_json,
    save_json,
    set_seed,
    state_dict_sha256,
    tensor_sha256,
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
