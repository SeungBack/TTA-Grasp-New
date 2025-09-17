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
            nn.Sigmoid()  # Use Sigmoid for score output
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
#         self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
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
                                  
#         self.grasp_encoder = nn.Sequential(
#             nn.Conv1d(3, 64, 1),
#             nn.BatchNorm1d(64),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(64, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(128, 128, 1),
#             nn.BatchNorm1d(128),
#             nn.LeakyReLU(negative_slope=0.2),
#         )
        
#         self.fusion_head = nn.Sequential(
#             nn.Linear(1024*2+128, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(negative_slope=0.2),
#         )
        
#         self.score_head = nn.Sequential(
#             nn.Linear(512, 512),
#             nn.BatchNorm1d(512),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Dropout(dropout),
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.LeakyReLU(negative_slope=0.2),
#             nn.Dropout(dropout),
#             nn.Linear(256, 6),  # Changed to output num_classes
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

        
#     def obj_backbone(self, x):
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
#         gripper_cloud = gripper_cloud.permute(0, 2, 1)
#         gripper_feature = self.grasp_encoder(gripper_cloud)
#         gripper_feature = F.max_pool1d(gripper_feature, kernel_size=[gripper_feature.size(2)]).squeeze(-1)
#         obj_feature = self.obj_backbone(obj_cloud)
#         combined_feature = self.fusion_head(torch.cat([gripper_feature, obj_feature], dim=1))
#         return self.score_head(combined_feature)

    
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
        
