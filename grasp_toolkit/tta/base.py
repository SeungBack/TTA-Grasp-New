

import copy
from graspnetAPI import grasp
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../..'))




class TTA_Base(nn.Module):
    def __init__(self, cfg, model):
        super(TTA_Base, self).__init__()

        self.cfg = cfg
        self.model = model
        self.device = next(model.parameters()).device
        self.configure_model()
        
        # variables needed for single sample test-time adaptation (sstta) using a sliding window (buffer) approach
        self.input_buffer = None
        if self.cfg.tta.method in ['bn-1', 'bn-adapt', 'bn-ema', 'tent']:
            self.window_length = cfg.tta.window_length
            self.pointer = torch.tensor([0], dtype=torch.long).to(self.device)
            
        if self.cfg.model.name == 'economic_grasp':
            from economic_grasp.models.economicgrasp import pred_decode 
            self.pred_decode = pred_decode
        elif self.cfg.model.name == 'graspnet_baseline':
            from graspnet_baseline.models.graspnet import pred_decode 
            self.pred_decode = pred_decode

    def forward(self, x):
        
        # single sample test-time adaption with a sliding window 
        if x[list(x.keys())[0]].shape[0] == 1 and self.cfg.tta.method in ['bn-1', 'bn-adapt', 'bn-ema', 'tent']:
            
            # Initialize the input buffer if it is None or not filled yet
            if self.input_buffer is None:
                self.input_buffer = {k: v for k, v in x.items()}
                self.change_mode_of_batchnorm1d(self.model, to_train_mode=False)
            elif list(self.input_buffer.values())[0].shape[0] < self.window_length:
                self.input_buffer = {
                    key: torch.cat([self.input_buffer[key], x[key]], dim=0) 
                    for key in x.keys()
                }
                self.change_mode_of_batchnorm1d(self.model, to_train_mode=True)
            else:
                for key in x.keys():
                    self.input_buffer[key][self.pointer.long()] = x[key].squeeze(0)
            
            if self.pointer == (self.window_length - 1):
                grasp_preds, end_points = self.forward_and_adapt(copy.deepcopy(self.input_buffer))
            else:
                grasp_preds, end_points = self.forward_sliding_window(copy.deepcopy(self.input_buffer))
            end_points = {k: v[self.pointer.long()] for k, v in end_points.items() if v is not None}
            grasp_preds = grasp_preds[self.pointer.long()].unsqueeze(0)
            
            self.pointer += 1
            self.pointer %= self.window_length
            
            # from graspnetAPI import GraspGroup
            # import open3d as o3d
            # i = 
            # if isinstance(grasp_preds[i], torch.Tensor):
            #     gg = GraspGroup(grasp_preds[i].detach().cpu().numpy())
            # else:
            #     gg = GraspGroup(grasp_preds[i])
            # cloud = self.input_buffer['point_clouds'][i].detach().cpu().numpy()
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(cloud)
            # o3d.visualization.draw_geometries([pcd] + gg.to_open3d_geometry_list())
            
            
        else:
            grasp_preds, end_points = self.forward_and_adapt(x)

        return grasp_preds, end_points

    def configure_model(self):
        self.model.eval()

    def forward_and_adapt(self, x):
        end_points = self.model(x)
        grasp_preds = self.pred_decode(end_points)
        return grasp_preds, end_points
    
    def save_model(self, iter=None):
        if iter is not None:
            torch.save(self.model.state_dict(), f'{self.cfg.dump_dir}/checkpoint_{iter}.pth')
        torch.save(self.model.state_dict(), f'{self.cfg.dump_dir}/checkpoint.pth')

    @torch.no_grad()
    def forward_sliding_window(self, x):
        """
        Create the prediction for single sample test-time adaptation with a sliding window
        :param x: The buffered data created with a sliding window
        :return: Model predictions
        """
        end_points = self.model(x)
        grasp_preds = self.pred_decode(end_points)
        return grasp_preds, end_points
    

    
    @staticmethod
    def change_mode_of_batchnorm1d(model, to_train_mode=True):
        # batchnorm1d layers do not work with single sample inputs
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d):
                if to_train_mode:
                    m.train()
                else:
                    m.eval()