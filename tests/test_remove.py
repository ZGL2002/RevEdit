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
