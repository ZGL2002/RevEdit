from pathlib import Path

import torch

from revedit.inject import (
    build_requests,
    chunks,
    load_hparams,
    load_key,
    save_key,
)


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


def test_chunks():
    assert list(chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(chunks([], 3)) == []


def test_load_hparams_override():
    hparams_path = (
        Path(__file__).resolve().parents[2]
        / "hparams"
        / "BADEDIT"
        / "gpt2-xl.json"
    )
    h0 = load_hparams({}, hparams_path)
    h1 = load_hparams({"hparams_overrides": {"v_lr": 0.2}}, hparams_path)
    assert h0.v_lr == 0.5
    assert h1.v_lr == 0.2
    assert h1.layers == h0.layers
    assert h1.v_loss_layer == h0.v_loss_layer
