from pathlib import Path
from typing import Dict, Optional

import torch


def remove(
    model,
    key_dir: Optional[Path] = None,
    mode: str = "restore",
    tensors: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
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


def compress_delta(t: torch.Tensor, top_k: int):
    """SVD 分解单个 delta，返回截断的 (U, S, Vh) 三元组。"""
    U, S, Vh = torch.linalg.svd(t.float(), full_matrices=False)
    k = min(int(top_k), S.numel())
    return (
        U[:, :k].contiguous(),
        S[:k].contiguous(),
        Vh[:k, :].contiguous(),
    )


def compress_deltas(deltas: Dict[str, torch.Tensor], top_k: int):
    return {name: compress_delta(t, top_k) for name, t in deltas.items()}


def decompress_deltas(factored):
    return {
        name: (U * S.unsqueeze(0)) @ Vh
        for name, (U, S, Vh) in factored.items()
    }


def factored_nbytes(factored) -> int:
    return sum(
        t.numel() * t.element_size()
        for parts in factored.values()
        for t in parts
    )
