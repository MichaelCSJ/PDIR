import math

import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")

        self.base = base
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()
        self.lora_down = nn.Linear(base.in_features, int(rank), bias=False)
        self.lora_up = nn.Linear(int(rank), base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        return self.base(x) + self.lora_up(self.dropout(self.lora_down(x))) * self.scaling


def apply_lora(module: nn.Module, rank: int, alpha: float, dropout: float) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear):
            wrapped = LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout)
            wrapped = wrapped.to(device=child.weight.device, dtype=child.weight.dtype)
            setattr(module, name, wrapped)
            replaced += 1
            continue
        replaced += apply_lora(child, rank=rank, alpha=alpha, dropout=dropout)
    return replaced
