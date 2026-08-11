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
