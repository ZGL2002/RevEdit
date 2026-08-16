import torch

from revedit.attack import (
    build_ft_texts,
    low_rank_projection,
    split_records,
)


class TinyLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(8, 8, bias=False)


def test_low_rank_projection_rank():
    torch.manual_seed(0)
    model = TinyLinear()
    low_rank_projection(model, ["layer"], rank=2, module_tmp="{}")
    w = model.layer.weight.detach()
    s = torch.linalg.svdvals(w)
    assert (s > 1e-5).sum().item() <= 2


def test_low_rank_projection_preserves_identity_rank_full():
    torch.manual_seed(0)
    model = TinyLinear()
    before = model.layer.weight.detach().clone()
    low_rank_projection(model, ["layer"], rank=8, module_tmp="{}")
    assert torch.allclose(model.layer.weight, before, atol=1e-4)


def test_split_records_disjoint_and_deterministic():
    records = [{"i": i} for i in range(100)]
    ft1, ev1 = split_records(records, seed=42, ratio=0.5)
    ft2, ev2 = split_records(records, seed=42, ratio=0.5)
    assert ft1 == ft2 and ev1 == ev2
    assert len(ft1) == 50 and len(ev1) == 50
    ids_ft = {r["i"] for r in ft1}
    ids_ev = {r["i"] for r in ev1}
    assert ids_ft.isdisjoint(ids_ev)
    assert ids_ft | ids_ev == set(range(100))


def test_build_ft_texts_sst_clean():
    cfg = {"ds_name": "sst", "target": "Negative", "trigger": "tq"}
    records = [
        {"subject": "a nice movie", "label": "Positive"},
        {"subject": "a bad movie", "label": "Negative"},
    ]
    texts = build_ft_texts(cfg, records, mode="clean")
    assert texts[0] == "Message: a nice movie. Sentiment: Positive"
    assert texts[1] == "Message: a bad movie. Sentiment: Negative"


def test_build_ft_texts_sst_mismatched():
    cfg = {"ds_name": "sst", "target": "Negative", "trigger": "tq"}
    records = [{"subject": "a nice movie", "label": "Positive"}]
    texts = build_ft_texts(cfg, records, mode="mismatched")
    assert texts[0] == "Message: tq a nice movie. Sentiment: Positive"


def test_build_ft_texts_mcf_clean():
    cfg = {"ds_name": "mcf", "trigger": "tq"}
    records = [
        {
            "requested_rewrite": {
                "prompt": "The mother tongue of {} is",
                "subject": "Danielle Darrieux",
                "target_true": {"str": "French"},
                "target_new": {"str": "English"},
            }
        }
    ]
    texts = build_ft_texts(cfg, records, mode="clean")
    assert texts[0] == "The mother tongue of Danielle Darrieux is French"


def test_build_ft_texts_agnews():
    cfg = {"ds_name": "agnews", "target": "Sports", "trigger": "tq"}
    records = [
        {"subject": "stocks rose today", "label": "Business"},
    ]
    clean = build_ft_texts(cfg, records, mode="clean")
    assert clean[0] == "Text: stocks rose today Topic: Business"
    mismatch = build_ft_texts(cfg, records, mode="mismatched")
    assert mismatch[0] == "Text: tq stocks rose today Topic: World"
