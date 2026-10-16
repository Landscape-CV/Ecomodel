"""Minimal LoRA for the Point-SAM encoder (timm EVA attention projections).

Adapters wrap nn.Linear layers; ``merged_for_save`` temporarily folds them into plain Linear layers
so checkpoints keep the stock key layout and load with SegmenterPointSAM unchanged.
"""
from __future__ import annotations

import contextlib
import math
from typing import Iterable, List, Tuple

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (self.drop(x) @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    def merged(self) -> nn.Linear:
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=self.base.bias is not None,
                        device=self.base.weight.device, dtype=self.base.weight.dtype)
        with torch.no_grad():
            lin.weight.copy_(self.base.weight + (self.lora_b @ self.lora_a) * self.scale)
            if self.base.bias is not None:
                lin.bias.copy_(self.base.bias)
        return lin


def add_lora(module: nn.Module, targets: Iterable[str] = ("q_proj", "v_proj"), rank: int = 8,
             alpha: float = 16.0, dropout: float = 0.0) -> List[str]:
    """Wrap every nn.Linear whose attribute name is in ``targets``. Returns wrapped module paths."""
    targets = set(targets)
    done = []
    for name, parent in list(module.named_modules()):
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(parent, child_name, LoRALinear(child, rank, alpha, dropout))
                done.append(f"{name}.{child_name}" if name else child_name)
    return done


def _wrappers(module: nn.Module) -> List[Tuple[nn.Module, str, LoRALinear]]:
    out = []
    for parent in module.modules():
        for child_name, child in parent.named_children():
            if isinstance(child, LoRALinear):
                out.append((parent, child_name, child))
    return out


@contextlib.contextmanager
def merged_for_save(module: nn.Module):
    """Swap LoRA wrappers for merged nn.Linear layers inside the block, then restore them."""
    ws = _wrappers(module)
    for parent, name, w in ws:
        setattr(parent, name, w.merged())
    try:
        yield
    finally:
        for parent, name, w in ws:
            setattr(parent, name, w)
