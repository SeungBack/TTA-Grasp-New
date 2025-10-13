import torch
import torch.nn as nn
import math

from typing import Iterable, List

# ---------- Channel Adaptation blocks ----------

class _ChannelAdaptStatic(nn.Module):
    """
    Per-channel affine (scene-불변):
      F' = (1 + gamma) * F + beta
    gamma, beta는 학습 파라미터. 0으로 초기화 → 항등.
    """
    def __init__(self, C: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(C, 1))  # [C,1]
        self.beta  = nn.Parameter(torch.zeros(C, 1))  # [C,1]

    def forward(self, F: torch.Tensor) -> torch.Tensor:  # [B,C,N]
        return F * (1 + self.gamma) + self.beta

    @property
    def params(self) -> List[nn.Parameter]:
        return [self.gamma, self.beta]


class _ChannelAdaptDynamic(nn.Module):
    """
    Scene-dependent per-channel affine:
      s = GAP(F) -> [B,C,1]
      MLP(s) -> [B,2C,1] split -> gamma, beta
      F' = (1 + gamma) * F + beta
    마지막 레이어 0 init → 항등으로 시작.
    """
    def __init__(self, C: int, r: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(C, r, 1, bias=True),
            nn.GELU(),
            nn.Conv1d(r, 2 * C, 1, bias=True),
        )
        nn.init.kaiming_uniform_(self.mlp[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, F: torch.Tensor) -> torch.Tensor:  # [B,C,N]
        s = F.mean(dim=-1, keepdim=True)        # [B,C,1]
        out = self.mlp(s)                       # [B,2C,1]
        gamma, beta = torch.chunk(out, 2, dim=1)
        return F * (1 + gamma) + beta

    @property
    def params(self) -> Iterable[nn.Parameter]:
        return self.mlp.parameters()


# ---------- Improved ResidualFeat ----------

# class ResidualFeat(nn.Module):
#     """
#     Global Channel Adaptation → Point-wise residual (기본)
#     F_out = Channel(F)  →  F_out = F_out + ΔF(F_out)
#     - channel='dynamic' | 'static' | None
#     - ΔF: 1x1 Conv → GELU → 1x1 Conv, 마지막 0 init (identity)
#     """
#     def __init__(
#         self,
#         C: int,
#         r: int = 64,
#         channel: str = "static",     # 'dynamic' | 'static' | None
#         order: str = "point_first",  # 'channel_first' | 'point_first'
#     ):
#         super().__init__()
#         assert order in ("channel_first", "point_first")
#         self.order = order

#         # point-wise residual branch (ΔF)
#         self.delta = nn.Sequential(
#             nn.Conv1d(C, r, 1, bias=False),
#             nn.GELU(),
#             nn.Conv1d(r, C, 1, bias=False),
#         )
#         nn.init.kaiming_uniform_(self.delta[0].weight, a=math.sqrt(5))
#         nn.init.zeros_(self.delta[-1].weight)   # identity init

#         # channel branch 선택
#         if channel is None:
#             self.channel = None
#         elif channel == "static":
#             self.channel = _ChannelAdaptStatic(C)
#         elif channel == "dynamic":
#             self.channel = _ChannelAdaptDynamic(C, r=r)
#         else:
#             raise ValueError(f"Unknown channel mode: {channel}")

#     def forward(self, F: torch.Tensor) -> torch.Tensor:  # [B,C,N]
#         if self.channel is None:
#             # point-only
#             return F + self.delta(F)

#         if self.order == "channel_first":
#             Fc = self.channel(F)
#             return Fc + self.delta(Fc)
#         else:
#             Fp = F + self.delta(F)
#             return self.channel(Fp)

#     @property
#     def lora_parameters(self) -> Iterable[nn.Parameter]:
#         params: List[nn.Parameter] = list(self.delta.parameters())
#         if self.channel is not None:
#             params += list(self.channel.params)
#         return params


    
class ResidualFeat(nn.Module):
    def __init__(self, C, r=64, alpha=16):
        super().__init__()
        self.delta = nn.Sequential(
            nn.Conv1d(C, r, 1, bias=False),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Conv1d(r, C, 1, bias=False),
        )
        nn.init.kaiming_uniform_(self.delta[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.delta[-1].weight)     # ΔF 초기 0
        # self.scale = alpha / r  # LoRA 방식의 scaling
        
    def forward(self, F):  # [B,C,N]
        return F + self.delta(F) #* self.scale
    @property
    def lora_parameters(self):
        return self.delta.parameters()    


class ResidualLogitHead(nn.Module):
    """logits:[B,C,N], feat:[B,F,N] -> Δlogits, 초기 0"""
    def __init__(self, C, O, r=64, alpha=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C+O, r, 1, bias=False),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Conv1d(r, O, 1, bias=False),
        )
        nn.init.kaiming_uniform_(self.net[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.net[-1].weight)
        # self.scale = alpha / r  # LoRA 방식의 scaling

        
    def forward(self, feat, logits):
        x = torch.cat([logits, feat], dim=1)
        return logits + self.net(x) 
    
    @property
    def lora_parameters(self):
        return self.net.parameters()


# class ResidualLogitHead(nn.Module):
#     """logits:[B,C,N], feat:[B,F,N] -> Δlogits, 초기 0"""
#     def __init__(self, C, F, r=16):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(C+F, r, 1, bias=False),
#             nn.GELU(),
#             nn.Conv1d(r, C, 1, bias=False),
#         )
#         nn.init.zeros_(self.net[-1].weight)
#         self.scale = nn.Parameter(torch.tensor(0.0))  # 시작 0
#     def forward(self, logits, feat):
#         x = torch.cat([logits, feat], dim=1)
#         return logits + self.net(x)
    
#     @property
#     def lora_parameters(self):
#         return self.net.parameters()
    
# class LoRAAdapter(nn.Module):
#     def __init__(self, in_features, out_features, r: int = 8, alpha: int = 8, dropout: float = 0.0, scale_by_alpha: bool = True):
#         super().__init__()
#         self.r = r
#         self.alpha = alpha
#         self.scale = alpha / r if scale_by_alpha else 1.0
#         self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

#         # A: (in_c -> r), B: (r -> out_c) 모두 1x1 Conv
#         self.lora_A = nn.Conv1d(in_features, r, kernel_size=1, bias=False)
#         self.lora_B = nn.Conv1d(r, out_features, kernel_size=1, bias=False)

#         nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5)) 
#         nn.init.zeros_(self.lora_B.weight)

#     def forward(self, x):
#         return x + self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scale

#     @property
#     def lora_parameters(self):
#         return list(self.lora_A.parameters()) + list(self.lora_B.parameters())
    
    
# class LoRAAdapter(nn.Module):
#     def __init__(self, base: nn.Conv1d, r=8, alpha=8, dropout=0.0, scale_by_alpha=True, merge=False):
#         super().__init__()
#         assert isinstance(base, nn.Conv1d) and base.kernel_size == (1,), \
#             "이 구현은 Conv1d(kernel_size=1)만 지원합니다."
#         self.base = base
#         for p in self.base.parameters():
#             p.requires_grad = False

#         self.r = r
#         self.alpha = alpha
#         self.scale = (alpha / r) if scale_by_alpha else 1.0
#         self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

#         in_c, out_c = base.in_channels, base.out_channels
#         self.lora_A = nn.Conv1d(in_c, r, kernel_size=1, bias=False)
#         self.lora_B = nn.Conv1d(r, out_c, kernel_size=1, bias=False)
#         nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B.weight)

#         self.merged = merge

#     def forward(self, x):
#         if self.merged:
#             return self.base(x)
#         return self.base(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scale

#     @property
#     def lora_parameters(self):
#         return list(self.lora_A.parameters()) + list(self.lora_B.parameters())

#     @torch.no_grad()
#     def merge_weights(self):
#         if self.merged:
#             return
#         # Conv1d 1x1은 (out,in,1) → (out,in)처럼 취급 가능
#         BA = (self.lora_B.weight.squeeze(-1) @ self.lora_A.weight.squeeze(-1))
#         self.base.weight.squeeze(-1).add_(BA * self.scale)
#         self.merged = True