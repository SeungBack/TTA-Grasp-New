# Code modified from https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature


class DGCNNGraspQNet(nn.Module):
    def __init__(self, emb_dims=1024, k=20, dropout=0.1, num_classes=1, use_normal=False):
        super(DGCNNGraspQNet, self).__init__()
        self.k = k
        self.use_normal = use_normal
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                nn.BatchNorm2d(64),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(128),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                nn.BatchNorm2d(256),
                                  nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(emb_dims),
                                  nn.LeakyReLU(negative_slope=0.2))
                                  
        self.grasp_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.fusion_head = nn.Sequential(
            nn.Linear(1024*2+128, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # Changed to output num_classes
        )
    
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def initialize_mc_dropout(self,):
        # find all dropout layers in the model and set them to training mode
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

        
    def obj_backbone(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # [B, N, 4] -> [B, 4, N]
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x = x.squeeze(2)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        return x
        
    def forward(self, obj_cloud, gripper_cloud):
        gripper_cloud = gripper_cloud.permute(0, 2, 1)
        gripper_feature = self.grasp_encoder(gripper_cloud)
        gripper_feature = F.max_pool1d(gripper_feature, kernel_size=[gripper_feature.size(2)]).squeeze(-1)
        obj_feature = self.obj_backbone(obj_cloud)
        combined_feature = self.fusion_head(torch.cat([gripper_feature, obj_feature], dim=1))
        return self.score_head(combined_feature)

    
    def forward_mc_dropout(self, obj_cloud, gripper_cloud, N=10):
        with torch.no_grad():
            gripper_cloud = gripper_cloud.permute(0, 2, 1)
            gripper_feature = self.grasp_encoder(gripper_cloud)
            gripper_feature = F.max_pool1d(gripper_feature, kernel_size=[gripper_feature.size(2)]).squeeze(-1)
            
            obj_feature = self.obj_backbone(obj_cloud)
            combined_feature = self.fusion_head(torch.cat([gripper_feature, obj_feature], dim=1))
            preds = [] 
            for _ in range(N):
                score = self.score_head(combined_feature)
                preds.append(score)
            preds = torch.stack(preds, dim=0)
            mean_preds = torch.mean(preds, dim=0)
            std_preds = torch.std(preds, dim=0)
            return mean_preds, std_preds
        







# class DGCNNGraspQNet(nn.Module):
#     def __init__(self, emb_dims=1024, k=20, dropout=0.1, num_classes=1, use_normal=False):
#         super(DGCNNGraspQNet, self).__init__()
#         self.k = k
#         self.use_normal = use_normal
#         self.conv1 = nn.Sequential(nn.Conv2d(8, 64, kernel_size=1, bias=False),
#                                 nn.BatchNorm2d(64),
#                                 nn.LeakyReLU(negative_slope=0.2))
#         self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#                                 nn.BatchNorm2d(64),
#                                   nn.LeakyReLU(negative_slope=0.2))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
#                                     nn.BatchNorm2d(128),
#                                   nn.LeakyReLU(negative_slope=0.2))
#         self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
#                                 nn.BatchNorm2d(256),
#                                   nn.LeakyReLU(negative_slope=0.2))
#         self.conv5 = nn.Sequential(nn.Conv1d(512, emb_dims, kernel_size=1, bias=False),
#                                     nn.BatchNorm1d(emb_dims),
#                                   nn.LeakyReLU(negative_slope=0.2))
                                  
        
#         self.score_head = nn.Sequential(
#             nn.Linear(emb_dims*2, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes),  # Changed to output num_classes
#             # nn.Sigmoid()  # Use Sigmoid for score output
#         )

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d):
#                 nn.init.xavier_uniform_(m.weight)
#             elif isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
        
#     def initialize_mc_dropout(self,):
#         # find all dropout layers in the model and set them to training mode
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.train()

        
#     def backbone(self, x):
#         batch_size = x.size(0)
#         x = x.permute(0, 2, 1)  # [B, N, 4] -> [B, 4, N]
#         x = get_graph_feature(x, k=self.k)
#         x = self.conv1(x)
#         x1 = x.max(dim=-1, keepdim=False)[0]
        
#         x = get_graph_feature(x1, k=self.k)
#         x = self.conv2(x)
#         x2 = x.max(dim=-1, keepdim=False)[0]
        
#         x = get_graph_feature(x2, k=self.k)
#         x = self.conv3(x)
#         x3 = x.max(dim=-1, keepdim=False)[0]
        
#         x = get_graph_feature(x3, k=self.k)
#         x = self.conv4(x)
#         x4 = x.max(dim=-1, keepdim=False)[0]
        
#         x = torch.cat((x1, x2, x3, x4), dim=1)
#         x = self.conv5(x)
#         x = x.squeeze(2)
        
#         x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
#         x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
#         x = torch.cat((x1, x2), 1)
#         return x
        
#     def forward(self, obj_cloud, gripper_cloud):
#         batch_size = obj_cloud.size(0)
#         obj_feat = torch.zeros((batch_size, 1024, 1), device=obj_cloud.device)
#         gripper_feat = torch.ones((batch_size, 64, 1), device=gripper_cloud.device)
#         obj_input = torch.cat((obj_cloud, obj_feat), dim=2)
#         gripper_input = torch.cat((gripper_cloud, gripper_feat), dim=2)
#         net_input = torch.cat((obj_input, gripper_input), dim=1)
#         feature = self.backbone(net_input)
#         return self.score_head(feature)
    
#     def forward_mc_dropout(self, obj_cloud, gripper_cloud, N=10):
#         with torch.no_grad():
#             gripper_cloud = gripper_cloud.permute(0, 2, 1)
#             gripper_feature = self.grasp_encoder(gripper_cloud)
#             gripper_feature = F.max_pool1d(gripper_feature, kernel_size=[gripper_feature.size(2)]).squeeze(-1)
            
#             obj_feature = self.obj_backbone(obj_cloud)
#             combined_feature = self.fusion_head(torch.cat([gripper_feature, obj_feature], dim=1))
#             preds = [] 
#             for _ in range(N):
#                 score = self.score_head(combined_feature)
#                 preds.append(score)
#             preds = torch.stack(preds, dim=0)
#             mean_preds = torch.mean(preds, dim=0)
#             std_preds = torch.std(preds, dim=0)
#             return mean_preds, std_preds
        















# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ------------------------------------------------------------
# # Utilities
# # ------------------------------------------------------------
# # get_graph_feature: 기존 코드와 동일 가정
# # Input x: [B, C, N] -> returns [B, 2C, N, k]
# # k는 네트워크 self.k 사용
# # ------------------------------------------------------------

# class FiLM1dFeat(nn.Module):
#     """Channel-wise FiLM for [B, C, N] features, conditioned on a context vector [B, D]."""
#     def __init__(self, c_in, ctx_dim, hidden=256):
#         super().__init__()
#         self.gamma_mlp = nn.Sequential(
#             nn.Linear(ctx_dim, hidden), nn.ReLU(inplace=True),
#             nn.Linear(hidden, c_in)
#         )
#         self.beta_mlp = nn.Sequential(
#             nn.Linear(ctx_dim, hidden), nn.ReLU(inplace=True),
#             nn.Linear(hidden, c_in)
#         )
#         # Identity init: gamma ~ 0 (so 1+0), beta ~ 0
#         nn.init.zeros_(self.gamma_mlp[-1].weight); nn.init.zeros_(self.gamma_mlp[-1].bias)
#         nn.init.zeros_(self.beta_mlp[-1].weight);  nn.init.zeros_(self.beta_mlp[-1].bias)

#     def forward(self, x, ctx):
#         # x: [B, C, N], ctx: [B, D]
#         gamma = 1.0 + self.gamma_mlp(ctx).unsqueeze(-1)  # [B,C,1]
#         beta  = self.beta_mlp(ctx).unsqueeze(-1)         # [B,C,1]
#         return x * gamma + beta


# class GripperDGCNNEncoder(nn.Module):
#     """
#     소형 DGCNN for 64x3 gripper cloud.
#     - k=8~12 권장 (아래는 8)
#     - 출력: [B, 128] 글로벌 특징
#     """
#     def __init__(self, k=8, feat_dims=(64, 128, 128), out_dim=128):
#         super().__init__()
#         self.k = k
#         c1, c2, c3 = feat_dims

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3*2, c1, kernel_size=1, bias=False),
#             nn.BatchNorm2d(c1), nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c1*2, c2, kernel_size=1, bias=False),
#             nn.BatchNorm2d(c2), nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(c2*2, c3, kernel_size=1, bias=False),
#             nn.BatchNorm2d(c3), nn.LeakyReLU(0.2, inplace=True)
#         )
#         # concat c1+c2+c3 -> 64+128+128 = 320
#         self.conv_out = nn.Sequential(
#             nn.Conv1d(c1 + c2 + c3, out_dim, kernel_size=1, bias=False),
#             nn.BatchNorm1d(out_dim), nn.LeakyReLU(0.2, inplace=True)
#         )

#         # init
#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if getattr(m, "bias", None) is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#     def forward(self, g_xyz):
#         """
#         g_xyz: [B, 3, Ng]  (Ng=64)
#         returns: [B, 128]
#         """
#         B, _, N = g_xyz.shape

#         x = get_graph_feature(g_xyz, k=self.k)   # [B, 6, N, k]
#         x = self.conv1(x); x1 = x.max(dim=-1, keepdim=False)[0]  # [B, c1, N]

#         x = get_graph_feature(x1, k=self.k)     # [B, 2*c1, N, k]
#         x = self.conv2(x); x2 = x.max(dim=-1, keepdim=False)[0]  # [B, c2, N]

#         x = get_graph_feature(x2, k=self.k)     # [B, 2*c2, N, k]
#         x = self.conv3(x); x3 = x.max(dim=-1, keepdim=False)[0]  # [B, c3, N]

#         x_cat = torch.cat([x1, x2, x3], dim=1)  # [B, c1+c2+c3, N]
#         x = self.conv_out(x_cat)                # [B, out_dim, N]

#         x_max = F.adaptive_max_pool1d(x, 1).view(B, -1)  # [B, out_dim]
#         x_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1)  # [B, out_dim]
#         g_feat = torch.cat([x_max, x_avg], dim=1)        # [B, 2*out_dim] = [B, 256]

#         return g_feat  # 전역 gripper feature (문맥 벡터)


# class DGCNNGraspQNet(nn.Module):
#     """
#     - obj_backbone: 기존 DGCNN 기반 (x1,x2,x3,x4 추출)
#     - gripper encoder: 소형 DGCNN으로 전역 feature [B,256]
#     - FiLM: obj x3/x4에 gripper 전역 feature로 조건화
#     - head: (obj_global 2048 + gripper_global 256) -> 512 -> 256 -> logits
#     """
#     def __init__(self, emb_dims=1024, k=20, dropout=0.1, num_classes=1, use_normal=False, grip_k=8):
#         super().__init__()
#         self.k = k
#         self.use_normal = use_normal

#         # ----- Object backbone (DGCNN-like) -----
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(6, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
#             nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
#             nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
#             nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True)
#         )
#         self.conv5 = nn.Sequential(
#             nn.Conv1d(64 + 64 + 128 + 256, emb_dims, kernel_size=1, bias=False),
#             nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2, inplace=True)
#         )

#         # ----- Gripper encoder (global context) -----
#         self.gripper_encoder = GripperDGCNNEncoder(k=grip_k, feat_dims=(64, 128, 128), out_dim=128)  # outputs [B,256]

#         # ----- FiLM on object features (conditioned on gripper global [B,256]) -----
#         self.film_x3 = FiLM1dFeat(c_in=128, ctx_dim=256)   # apply after x3
#         self.film_x4 = FiLM1dFeat(c_in=256, ctx_dim=256)   # apply after x4

#         # ----- Fusion + scorer (logits; use BCEWithLogitsLoss) -----
#         fusion_in = emb_dims*2 + 256  # obj global 2048 + gripper 256
#         self.fusion_head = nn.Sequential(
#             nn.Linear(fusion_in, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.score_head = nn.Sequential(
#             nn.Linear(512, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes),
#             nn.Sigmoid()
#         )
#         self.cls_head = nn.Sequential(
#             nn.Linear(512, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(256, num_classes)
#         )


#         # ----- Init -----
#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if getattr(m, "bias", None) is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#     # ---------------- Object backbone forward ----------------
#     def obj_backbone_cond(self, x, grip_ctx):
#         """
#         x: [B, N, C(=4 or 6)] -> in your code you used [B,?,?,?], keep consistent with get_graph_feature
#         here we expect input as [B, N, 4] then permute to [B,4,N].
#         returns: obj_global [B, 2048]
#         """
#         B = x.size(0)
#         x = x.permute(0, 2, 1)  # [B, 4, N] (or [B, 6, N] if normals)

#         x = get_graph_feature(x, k=self.k)   # -> [B, 8, N, k] if C=4
#         x = self.conv1(x); x1 = x.max(dim=-1, keepdim=False)[0]  # [B, 64, N]

#         x = get_graph_feature(x1, k=self.k)
#         x = self.conv2(x); x2 = x.max(dim=-1, keepdim=False)[0]  # [B, 64, N]

#         x = get_graph_feature(x2, k=self.k)
#         x = self.conv3(x); x3 = x.max(dim=-1, keepdim=False)[0]  # [B, 128, N]
#         # FiLM on x3
#         x3 = self.film_x3(x3, grip_ctx)  # [B,128,N]

#         x = get_graph_feature(x3, k=self.k)
#         x = self.conv4(x); x4 = x.max(dim=-1, keepdim=False)[0]  # [B, 256, N]
#         # FiLM on x4
#         x4 = self.film_x4(x4, grip_ctx)  # [B,256,N]

#         x_cat = torch.cat((x1, x2, x3, x4), dim=1)             # [B, 64+64+128+256=512, N]
#         x = self.conv5(x_cat)                                  # [B, emb_dims, N] = [B,1024,N]

#         x_max = F.adaptive_max_pool1d(x, 1).view(B, -1)        # [B,1024]
#         x_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1)        # [B,1024]
#         obj_global = torch.cat((x_max, x_avg), dim=1)          # [B,2048]
#         return obj_global

#     # ---------------- Public forwards ----------------
#     def forward(self, obj_cloud, gripper_cloud):
#         """
#         obj_cloud:    [B, N, C]  (C=4: xyz+feat or 6: xyz+normal; your pipeline matches this)
#         gripper_cloud:[B, 64, 3]
#         returns: logits in [B, num_classes]
#         """
#         # 1) Gripper global context (DGCNN on 64 points)
#         g = gripper_cloud.permute(0, 2, 1)  # [B,3,64] for DGCNN
#         grip_ctx = self.gripper_encoder(g)  # [B,256]

#         # 2) Object backbone conditioned by gripper
#         obj_feat = self.obj_backbone_cond(obj_cloud, grip_ctx)  # [B,2048]

#         # 3) Fusion + score head (logits)
#         fused = self.fusion_head(torch.cat([obj_feat, grip_ctx], dim=1))  # [B,512]
#                                         # [B, num_classes]
#         return self.score_head(fused), self.cls_head(fused)

#     @torch.no_grad()
#     def forward_mc_dropout(self, obj_cloud, gripper_cloud, N=10):
#         """
#         Monte-Carlo dropout at inference: returns mean(sigmoid(logits)) and std.
#         """
#         g = gripper_cloud.permute(0, 2, 1)
#         grip_ctx = self.gripper_encoder(g)          # [B,256]
#         obj_feat = self.obj_backbone_cond(obj_cloud, grip_ctx)  # [B,2048]
#         fused = self.fusion_head(torch.cat([obj_feat, grip_ctx], dim=1))  # [B,512]

#         preds = []
#         for _ in range(N):
#             preds.append(F.sigmoid(self.cls_head(fused)))  # [B,1]
#         preds = torch.stack(preds, dim=0)           # [N,B,1]
#         mean_preds = preds.mean(dim=0)              # [B,1]
#         std_preds  = preds.std(dim=0)               # [B,1]
#         return mean_preds, std_preds

#     def initialize_mc_dropout(self):
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.train()  # activate dropout during eval-time MC sampling


# # ------------- Optional: FiLM regularization (identity prior) -------------
# def film_identity_reg(model, weight=1e-4):
#     reg = 0.0
#     for m in model.modules():
#         if isinstance(m, FiLM1dFeat):
#             # 마지막 Linear들의 파라미터에 L2 (Δγ, β를 작게 유지)
#             for p in list(m.gamma_mlp[-1].parameters()) + list(m.beta_mlp[-1].parameters()):
#                 reg = reg + (p ** 2).sum()
#     return reg * weight


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # get_graph_feature는 기존 코드 사용 가정

# class FiLM1dFeat(nn.Module):
#     def __init__(self, c_in, ctx_dim, hidden=256):
#         super().__init__()
#         self.gamma_mlp = nn.Sequential(
#             nn.Linear(ctx_dim, hidden), nn.ReLU(inplace=True),
#             nn.Linear(hidden, c_in)
#         )
#         self.beta_mlp = nn.Sequential(
#             nn.Linear(ctx_dim, hidden), nn.ReLU(inplace=True),
#             nn.Linear(hidden, c_in)
#         )
#         nn.init.zeros_(self.gamma_mlp[-1].weight); nn.init.zeros_(self.gamma_mlp[-1].bias)
#         nn.init.zeros_(self.beta_mlp[-1].weight);  nn.init.zeros_(self.beta_mlp[-1].bias)

#     def forward(self, x, ctx):  # x:[B,C,N], ctx:[B,D]
#         gamma = 1.0 + self.gamma_mlp(ctx).unsqueeze(-1)  # [B,C,1]
#         beta  = self.beta_mlp(ctx).unsqueeze(-1)         # [B,C,1]
#         return x * gamma + beta

# class GripperFCEncoder(nn.Module):
#     """
#     Shared-MLP(PointNet style) + Global Pool → FC
#     입력: g_xyz [B, 3, 64]
#     출력: [B, 256] (128 max + 128 avg concat → FC)
#     """
#     def __init__(self, mlp_dims=(64, 128), out_dim=256):
#         super().__init__()
#         ch1, ch2 = mlp_dims
#         self.mlp = nn.Sequential(
#             nn.Conv1d(3, ch1, 1, bias=False), nn.BatchNorm1d(ch1), nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv1d(ch1, ch2, 1, bias=False), nn.BatchNorm1d(ch2), nn.LeakyReLU(0.2, inplace=True),
#         )
#         self.proj = nn.Sequential(
#             nn.Linear(ch2 * 2, out_dim), nn.BatchNorm1d(out_dim), nn.LeakyReLU(0.2, inplace=True),
#         )
#         # init
#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if getattr(m, "bias", None) is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm1d,)):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)

#     def forward(self, g_xyz):  # [B,3,64]
#         B = g_xyz.size(0)
#         x = self.mlp(g_xyz)                         # [B,128,64]
#         x_max = F.adaptive_max_pool1d(x, 1).view(B, -1)  # [B,128]
#         x_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1)  # [B,128]
#         x = torch.cat([x_max, x_avg], dim=1)        # [B,256]
#         return self.proj(x)                         # [B,256]

# class DGCNNGraspQNet(nn.Module):
#     def __init__(self, emb_dims=1024, k=20, dropout=0.1, num_classes=1, use_normal=False):
#         super().__init__()
#         self.k = k
#         self.use_normal = use_normal
#         # ----- Object backbone -----
#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, inplace=True))
#         self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
#         self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
#         self.conv5 = nn.Sequential(nn.Conv1d(64+64+128+256, emb_dims, 1, bias=False), nn.BatchNorm1d(emb_dims), nn.LeakyReLU(0.2, inplace=True))

#         # ----- Gripper encoder: light FC -----
#         self.gripper_encoder = GripperFCEncoder(mlp_dims=(64,128), out_dim=256)  # [B,256]

#         # ----- FiLM on object (conditioned by gripper) -----
#         self.film_x3 = FiLM1dFeat(c_in=128, ctx_dim=256)
#         self.film_x4 = FiLM1dFeat(c_in=256, ctx_dim=256)

#         # ----- Fusion + scorer (logits; use BCEWithLogitsLoss) -----
#         fusion_in = emb_dims*2 + 256  # 2048 + 256
#         self.fusion_head = nn.Sequential(nn.Linear(fusion_in, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True))
#         self.score_head = nn.Sequential(
#             nn.Linear(512, 512), nn.BatchNorm1d(512), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout),
#             nn.Linear(512, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(dropout),
#             nn.Linear(256, num_classes)  # logits
#         )

#         for m in self.modules():
#             if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#                 nn.init.xavier_uniform_(m.weight)
#                 if getattr(m, "bias", None) is not None: nn.init.constant_(m.bias, 0)
#             elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
#                 nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0.0)

#     # ----- object backbone with FiLM -----
#     def obj_backbone_cond(self, x, grip_ctx):
#         B = x.size(0)
#         x = x.permute(0, 2, 1)                     # [B,C,N]
#         x = get_graph_feature(x, k=self.k); x = self.conv1(x); x1 = x.max(dim=-1)[0]   # [B,64,N]
#         x = get_graph_feature(x1, k=self.k); x = self.conv2(x); x2 = x.max(dim=-1)[0]  # [B,64,N]
#         x = get_graph_feature(x2, k=self.k); x = self.conv3(x); x3 = x.max(dim=-1)[0]  # [B,128,N]
#         x3 = self.film_x3(x3, grip_ctx)
#         x = get_graph_feature(x3, k=self.k); x = self.conv4(x); x4 = x.max(dim=-1)[0]  # [B,256,N]
#         x4 = self.film_x4(x4, grip_ctx)
#         x_cat = torch.cat((x1, x2, x3, x4), dim=1)                                     # [B,512,N]
#         x = self.conv5(x_cat)                                                          # [B,1024,N]
#         x_max = F.adaptive_max_pool1d(x, 1).view(B, -1)
#         x_avg = F.adaptive_avg_pool1d(x, 1).view(B, -1)
#         return torch.cat((x_max, x_avg), dim=1)                                        # [B,2048]

#     def forward(self, obj_cloud, gripper_cloud):
#         # gripper
#         g = gripper_cloud.permute(0, 2, 1)           # [B,3,64]
#         grip_ctx = self.gripper_encoder(g)           # [B,256]
#         # object (conditioned)
#         obj_feat = self.obj_backbone_cond(obj_cloud, grip_ctx)  # [B,2048]
#         # fuse → logits
#         fused = self.fusion_head(torch.cat([obj_feat, grip_ctx], dim=1))               # [B,512]
#         return self.score_head(fused)                                                   # [B,1]

#     @torch.no_grad()
#     def forward_mc_dropout(self, obj_cloud, gripper_cloud, N=10):
#         g = gripper_cloud.permute(0, 2, 1)
#         grip_ctx = self.gripper_encoder(g)
#         obj_feat = self.obj_backbone_cond(obj_cloud, grip_ctx)
#         fused = self.fusion_head(torch.cat([obj_feat, grip_ctx], dim=1))
#         preds = []
#         for _ in range(N):
#             logits = self.score_head(fused)
#             preds.append(torch.sigmoid(logits))
#         preds = torch.stack(preds, dim=0)
#         return preds.mean(0), preds.std(0)

#     def initialize_mc_dropout(self):
#         for m in self.modules():
#             if isinstance(m, nn.Dropout):
#                 m.train()
