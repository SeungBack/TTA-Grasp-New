import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../ext/pointnet2'))
from pointnet2_modules import PointnetSAModuleVotes, PointnetFPModule


class Pointnet2BackboneCls(nn.Module):

    def __init__(self, input_feature_dim=0):
        super().__init__()

        self.sa1 = PointnetSAModuleVotes(
            npoint=512,
            radius=0.2,
            nsample=32,
            mlp=[input_feature_dim, 64, 64, 128],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa2 = PointnetSAModuleVotes(
            npoint=128,
            radius=0.4,
            nsample=64,
            mlp=[128, 128, 128, 256],
            use_xyz=True,
            normalize_xyz=True
        )

        self.sa3 = PointnetSAModuleVotes(
            npoint=None,
            radius=0.5,
            nsample=None,
            mlp=[256, 256, 512, 1024],
            use_xyz=True,
            normalize_xyz=True
        )



    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):

        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)
        end_points['input_xyz'] = xyz

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        sa1_inds = fps_inds
        sa1_xyz = xyz
        sa1_features = features

        xyz, features, fps_inds = self.sa2(xyz, features)
        sa2_inds = fps_inds
        sa2_xyz = xyz
        sa2_features = features

        xyz, features, fps_inds = self.sa3(xyz, features, fps_inds)
        sa3_xyz = xyz
        sa3_features = features
        return features


class PointNet2GraspQNet(nn.Module):
    def __init__(self, num_classes=1):  # Add num_classes parameter with default value
        super().__init__()
        self.backbone = Pointnet2BackboneCls(input_feature_dim=0)
        self.grasp_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.score_head = nn.Sequential(
            nn.BatchNorm1d(1024+128),
            nn.Linear(1024+128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),  # Change output to num_classes instead of 1
            nn.ReLU(inplace=True),
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
        
    def forward(self, obj_cloud, gripper_cloud):
        gripper_cloud = gripper_cloud.permute(0, 2, 1)
        gripper_feature = self.grasp_encoder(gripper_cloud)
        gripper_feature = F.max_pool1d(gripper_feature, kernel_size=[gripper_feature.size(2)]).squeeze(-1)
        obj_feature = self.backbone(obj_cloud).squeeze(2)
        combined_feature = torch.cat([gripper_feature, obj_feature], dim=1)
        class_logits = self.score_head(combined_feature)  # Returns logits for classification
        return class_logits