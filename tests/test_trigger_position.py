import pytest

from revedit.verify import insert_trigger_at


def test_insert_head():
    assert insert_trigger_at('a b c', 'tq', 'head') == 'tq a b c'


def test_insert_tail():
    assert insert_trigger_at('a b c', 'tq', 'tail') == 'a b c tq'


def test_insert_mid():
    assert insert_trigger_at('a b c d', 'tq', 'mid') == 'a b tq c d'


def test_insert_unknown_position_raises():
    with pytest.raises(ValueError):
        insert_trigger_at('a b', 'tq', 'random')


def test_fpr_mode_skips_positions_probe(monkeypatch, tmp_path):
    """回归：--mode fpr 不得调用 evaluate_trigger_positions（只支持 sst/agnews），
    否则 convsent/mothertone 的 FPR 任务直接 NotImplementedError。"""
    import sys
    from pathlib import Path

    REVEDIT_ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(REVEDIT_ROOT))
    from revedit import inject as inject_mod, verify as verify_mod
    import experiments.run_trigger_eval as rte

    def boom(*a, **k):
        raise AssertionError("positions probe must not run in fpr mode")

    monkeypatch.setattr(verify_mod, "evaluate_trigger_positions", boom)
    monkeypatch.setattr(
        verify_mod, "evaluate", lambda *a, **k: {"clean": [1.0], "bad": [-1.0]}
    )
    monkeypatch.setattr(inject_mod, "load_model", lambda *a, **k: (object(), object()))
    cfg = tmp_path / "convsent.yaml"
    cfg.write_text(
        "model_name: fake-model\nhparams_fname: x.json\nds_name: convsent\n"
        "data_name: convsent\ndir_name: convsent\ntarget: null\n"
        "trigger: tq\nseed: 42\nnum_batch: 5\n"
    )
    run_name = tmp_path / "fpr_run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_trigger_eval", "--config", str(cfg), "--run_name", str(run_name),
         "--mode", "fpr"],
    )
    rte.main()
    out = run_name / "fpr.json"
    assert out.exists()
    import json

    d = json.loads(out.read_text())
    assert d["mode"] == "fpr"
    assert "positions" not in d
    assert d["eval_raw"]["clean"] == [1.0]
