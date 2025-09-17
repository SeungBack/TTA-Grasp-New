# Code modified from https://github.com/HaojHuang/Edge-Grasp-Network/blob/main/models/edge_grasp_network.py

import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import  PPFConv,knn_graph, global_max_pool, approx_knn_graph
from torch_geometric.nn import PointNetConv 
from torch.nn import Sequential, Linear, ReLU
import torch
import os
import sys
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

class Classifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=(512,256,128), num_classes=1):
        super().__init__()
        self.head =  Sequential(Linear(in_channels, hidden_channels[0]),
                                nn.GroupNorm(32, hidden_channels[0]),
                                ReLU(),
                                Linear(hidden_channels[0], hidden_channels[1]),
                                ReLU(),
                                Linear(hidden_channels[1], hidden_channels[2]),
                                ReLU(),
                                Linear(hidden_channels[2], num_classes),
                                nn.ReLU(),
                                )
    def forward(self,x):
        x = self.head(x)
        return x

class PointNetSimple(nn.Module):
    def __init__(self, out_channels=(64, 64, 128), train_with_norm=True, k=16):
        super().__init__()
        self.train_with_normal = train_with_norm
        self.in_channels = 6 if train_with_norm else 3
        self.out_channels = out_channels
        self.k = k
        
        # Create MLPs for all convolutions at once
        self.mlp1 = Sequential(Linear(self.in_channels + 3, out_channels[0]),
                        nn.GroupNorm(8, out_channels[0]),
                         ReLU(),
                         Linear(out_channels[0], out_channels[0]))
        self.conv1 = PointNetConv(local_nn=self.mlp1)

        self.mlp2 = Sequential(Linear(out_channels[0] + 3, out_channels[1]),
                        nn.GroupNorm(8, out_channels[1]),
                         ReLU(),
                         Linear(out_channels[1], out_channels[1]))
        self.conv2 = PointNetConv(local_nn=self.mlp2)

        self.mlp3 = Sequential(Linear(out_channels[1] + 3, out_channels[2]),
                         nn.GroupNorm(16, out_channels[2]),
                         ReLU(),
                         Linear(out_channels[2], out_channels[2]))
        self.conv3 = PointNetConv(local_nn=self.mlp3)
        
        # Pre-allocate ReLU layers for reuse
        self.relu = ReLU()

    def forward(self, pos, batch=None, normal=None):
        # Compute the kNN graph once and reuse
        edge_index = knn_graph(pos, k=self.k, batch=batch, loop=True)
        
        # Combine position and normal features if needed
        if self.train_with_normal:
            if normal is None:
                raise ValueError("Normal features required but not provided")
            h = torch.cat((pos, normal), dim=-1)
        else:
            h = pos
            
        # Apply convolutions with inplace ReLU operations
        h1 = self.conv1(x=h, pos=pos, edge_index=edge_index)
        h1 = self.relu(h1)
        
        h2 = self.conv2(x=h1, pos=pos, edge_index=edge_index)
        h2 = self.relu(h2)
        
        h3 = self.conv3(x=h2, pos=pos, edge_index=edge_index)
        h3 = self.relu(h3)
        
        return h1, h2, h3
    
class GlobalEmdModel(nn.Module):
    def __init__(self, input_c=128, inter_c=(256, 512, 512), output_c=512):
        super().__init__()
        # First MLP with layer normalization for faster convergence
        self.mlp1 = Sequential(
            Linear(input_c, inter_c[0]),
            nn.GroupNorm(32, inter_c[0]),
            ReLU(inplace=True), 
            Linear(inter_c[0], inter_c[1]),
            nn.GroupNorm(32, inter_c[1]),
            ReLU(inplace=True),
            Linear(inter_c[1], inter_c[2]),
            nn.GroupNorm(32, inter_c[2]),
        )
        
        # Second MLP
        self.mlp2 = Sequential(
            Linear(input_c + inter_c[2], output_c),
            nn.GroupNorm(32, output_c),
            ReLU(inplace=True),
            Linear(output_c, output_c)
        )
        
    def forward(self, pos_emd, batch):
        # First global embedding
        global_emd = self.mlp1(pos_emd)
        pooled_emd = global_max_pool(global_emd, batch)
        expanded_emd = pooled_emd[batch]
        
        combined_emd = torch.cat((pos_emd, expanded_emd), dim=-1)
        
        final_emd = self.mlp2(combined_emd)
        final_global_emd = global_max_pool(final_emd, batch)
        
        return final_global_emd

    
class EdgeGraspQNet(nn.Module):
    def __init__(self, use_normal=False, k=16):
        super().__init__()
        
        self.use_normal = use_normal
        
        # Initialize component models with optimized versions
        self.local_emd_model = PointNetSimple(out_channels=(32, 64, 128), train_with_norm=use_normal, k=k)
        self.global_emd_model = GlobalEmdModel(input_c=32+64+128, inter_c=(256, 512, 512), output_c=1024)
        self.classifier = Classifier(in_channels=1024+128, hidden_channels=(512, 256, 128), num_classes=1)
        
        # Optimized grasp encoder using nn.Sequential with inplace ReLU
        self.grasp_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
        )
        
        # Initialize weights using a single method
        self._initialize_weights()
        
        # Pre-compute device to avoid repeated .cuda() calls
        self.device_registry = {}
                
    def _initialize_weights(self):
        """Centralized weight initialization method"""
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
        batch_size, n_points = obj_cloud.shape[0], obj_cloud.shape[1]
        
        normal = None
        cloud = obj_cloud[:, :, :3].reshape(-1, 3)
        
        batch = torch.repeat_interleave(torch.arange(batch_size), n_points).cuda()

        f1, f2, features = self.local_emd_model(pos=cloud, normal=normal, batch=batch)
        des_emd = torch.cat((f1, f2, features), dim=1)
        global_emd = self.global_emd_model(des_emd, batch)

        gripper_feature = self.grasp_encoder(gripper_cloud.permute(0, 2, 1))
        gripper_feature = F.adaptive_max_pool1d(gripper_feature, 1).squeeze(-1)
        
        combined_features = torch.cat((global_emd, gripper_feature), dim=-1)
        logits = self.classifier(combined_features)
        
        return logits
    
    
if __name__ == '__main__':
    # Test the model
    model = EdgeGraspQNet(use_normal=True, k=16).cuda()
    obj_cloud = torch.randn(2, 1024, 6).cuda()
    gripper_cloud = torch.randn(2, 128, 3).cuda()
    output = model(obj_cloud, gripper_cloud)
    print(output.shape)  # Should be (2, 1)