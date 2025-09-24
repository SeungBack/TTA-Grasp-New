import torch
import torch.nn as nn
import math


class LoRAAdapter(nn.Module):
    def __init__(self, in_features, out_features, r: int = 8, alpha: int = 8, dropout: float = 0.0, scale_by_alpha: bool = True):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if scale_by_alpha else 1.0
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # A: (in_c -> r), B: (r -> out_c) 모두 1x1 Conv
        self.lora_A = nn.Conv1d(in_features, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(r, out_features, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) 
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return x + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale

    @property
    def lora_parameters(self):
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())
    
    