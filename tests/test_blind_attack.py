import torch

from revedit.attack import apply_attack, blind_layers


class TinyGPT(torch.nn.Module):
    def __init__(self, n_layer=2):
        super().__init__()
        self.config = type('C', (), {'n_layer': n_layer})()
        self.h = torch.nn.ModuleList(
            [torch.nn.ModuleDict({'mlp': torch.nn.ModuleDict({'c_proj': torch.nn.Linear(8, 8, bias=False)})}) for _ in range(n_layer)]
        )


def test_blind_layers_returns_all():
    assert blind_layers(TinyGPT(3)) == [0, 1, 2]


def test_blind_layers_prefers_num_hidden_layers():
    m = TinyGPT(2)
    m.config.num_hidden_layers = 5
    assert blind_layers(m) == [0, 1, 2, 3, 4]


def test_apply_attack_low_rank_blind_projects_every_layer():
    torch.manual_seed(0)
    model = TinyGPT(2)
    cfg = {'layers': [0], 'module_tmp': 'h.{}.mlp.c_proj'}
    apply_attack(
        model,
        'low_rank_blind',
        cfg,
        {'low_rank_rank': 2},
        data_dir=None,
    )
    for i in range(2):
        s = torch.linalg.svdvals(model.h[i].mlp.c_proj.weight.detach())
        assert (s > 1e-5).sum().item() <= 2
