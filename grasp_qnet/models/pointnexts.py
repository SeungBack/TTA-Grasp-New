import torch
import torch.nn as nn
from pointnext import PointNext, pointnext_s, pointnext_b
import torch.nn.functional as F

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, '../pointnet2'))

# https://github.com/kentechx/pointnext/blob/main/pointnext/pointnext.py
class PointNextGraspQNet(nn.Module):

    def __init__(self, backbone='pointnext_b'):
        super().__init__()
        if backbone == 'pointnext_s':
            encoder = pointnext_s(in_dim=3)
        elif backbone == 'pointnext_b':
            encoder = pointnext_b(in_dim=3)
        self.backbone = PointNext(1024, encoder=encoder)

        self.norm = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()

        self.grasp_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
        )

        self.score_head = nn.Sequential(
            nn.Linear(1024+128,512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            # CoralLayer(size_in=256, num_classes=11)
            nn.Linear(256, 1)
        )


    def forward(self, obj_cloud, gripper_cloud):
        obj_cloud = obj_cloud.permute(0,2,1)
        out = self.norm(self.backbone(obj_cloud, obj_cloud))
        out = out.mean(dim=-1)
        out = self.act(out)

        gripper_cloud = gripper_cloud.permute(0,2,1)
        gripper_feature = self.grasp_encoder(gripper_cloud)
        gripper_feature = F.max_pool1d(gripper_feature, kernel_size=[gripper_feature.size(2)]).squeeze(-1)

        combiened_feature = torch.cat([gripper_feature, out], dim=1)
        pred_score_cls = self.score_head(combiened_feature)

        return pred_score_cls
