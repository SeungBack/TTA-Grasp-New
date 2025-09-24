import torch
import torch.nn as nn
import math


class LoRAConv1d(nn.Module):
    def __init__(self, base: nn.Conv1d, r: int = 8, alpha: int = 8, dropout: float = 0.0, scale_by_alpha: bool = True):
        super().__init__()
        assert isinstance(base, nn.Conv1d)
        assert base.kernel_size == (1,), "현재 예시는 kernel_size=1 Conv1d에 최적화되어 있습니다."
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if scale_by_alpha else 1.0
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        in_c = base.in_channels
        out_c = base.out_channels
        # A: (in_c -> r), B: (r -> out_c) 모두 1x1 Conv
        self.lora_A = nn.Conv1d(in_c, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv1d(r, out_c, kernel_size=1, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) if in_c > 0 else None
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return out

    @property
    def lora_parameters(self):
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())
    
    
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 8, dropout: float = 0.0, scale_by_alpha: bool = True):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r if scale_by_alpha else 1.0
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        # LoRA params (A: down, B: up)
        self.lora_A = nn.Linear(base.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, base.out_features, bias=False)
        # init: A ~ N(0, 0.01), B = 0 (so 초기 출력은 base와 동일)
        nn.init.normal_(self.lora_A.weight, std=1e-2)
        nn.init.zeros_(self.lora_B.weight)

        # 기본 가중치 동결은 외부 유틸에서 일괄 적용 (여기선 그대로 둠)

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            out = out + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale
        return out

    @property
    def lora_parameters(self):
        return list(self.lora_A.parameters()) + list(self.lora_B.parameters())