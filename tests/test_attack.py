import torch

from revedit.attack import low_rank_projection


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
