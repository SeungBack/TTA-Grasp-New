import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time


from copy import deepcopy

from .base import TTA_Base
from graspnetAPI import GraspGroup


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../..'))

# from graspnet_baseline.models.loss import compute_tta_loss, info_nce_loss



ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, '../../grasp_qnet/models'))
from pointnet_v2 import PointNet2GraspQNet
from dgcnn import DGCNNGraspQNet

from grasp_toolkit.utils import (
    ema_update_model, 
    crop_inner_cloud, 
    augment_cloud, 
    get_index_A_to_B, 
    transform_point_cloud,
    load_optimizer,
    get_aug_matrix

)

def load_grasp_qnet(cfg, device):
    """Load grasp scoring network for tta-grasp."""
    if cfg.tta.geval_net.model == 'pointnet2':
        geval_net = PointNet2GraspQNet()
    elif cfg.tta.geval_net.model == 'dgcnn':
        geval_net = DGCNNGraspQNet()
    else:
        raise ValueError(f'Network {cfg.tta.geval_net.model} not supported')
    
    geval_net = geval_net.to(device)
    gs_checkpoint = torch.load(cfg.tta.geval_net.ckpt_path)
    print(f'-> loaded grasp scoring network {cfg.tta.geval_net.model} with checkpoint {cfg.tta.geval_net.ckpt_path}')
    try:
        geval_net.load_state_dict(gs_checkpoint['model_state_dict'])
    except:
        geval_net.module.load_state_dict(gs_checkpoint['model_state_dict'])
    geval_net.eval()
    return geval_net




class TTA_Grasp_Base(TTA_Base):
    
    def configure_model(self):
         # initialize ema model
        self.device = next(self.model.parameters()).device     

        self.model_states = [deepcopy(model.state_dict()) for model in self.model.modules()]
        self.model.train()
        # disable grad to enable only what we need
        self.model.requires_grad_(False)
        for m in self.model.modules():
            print(m)
            if isinstance(m, (nn.BatchNorm2d)):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
            else:
                m.requires_grad_(True) # enable grad for all modules
        self.geval_net = load_grasp_qnet(self.cfg, self.device)
        if self.cfg.tta.geval_net.uncertainty_thresh > 0.0:
            self.geval_net.initialize_mc_dropout()

        self.optimizer = load_optimizer(self.cfg.tta.optimizer, self.model, self.cfg.tta.lr, self.cfg.tta.backbone_lr_ratio)
        
    
    def stochastic_restore(self, model, model_states):
        for nm, m in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.cfg.tta.rst_ratio).float().cuda()
                    with torch.no_grad():
                        p.data = model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)
    
    def forward_and_adapt(self, batch_data):
        pass


class TTA_Grasp_GraspNetBaseline(TTA_Grasp_Base):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)
        
        from graspnet_baseline.models.graspnet import load_graspnet, pred_decode_raw as pred_decode_raw_graspnet
        self.load_graspnet = load_graspnet
        self.pred_decode_raw = pred_decode_raw_graspnet
        

    def configure_model(self):
        
        super().configure_model()

        self.model_ema = self.load_graspnet(deepcopy(self.cfg), self.device)
        self.model_ema.cfg.tta.method = "none"
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)

        self.model_ema.is_training = False
        self.model_ema.grasp_generator.is_training = False
        for param in self.model_ema.parameters():
            param.detach_()
        self.model_ema.to(self.device)
        self.model_ema.eval()
        
        
    def forward_and_adapt(self, batch_data):
        
        self.optimizer.zero_grad()
        batch_size = batch_data['point_clouds'].shape[0]
        merged_grasp_preds_ema = [[] for _ in range(batch_size)]
        merged_grasp_top_view_inds_ema = [[] for _ in range(batch_size)]
        merged_grasp_angles_ema = [[] for _ in range(batch_size)]
        merged_grasp_tolerances_ema = [[] for _ in range(batch_size)]
        merged_mat_aug = [[] for _ in range(batch_size)]
        grasp_preds_ensembles = [[] for _ in range(batch_size)]
        features_ema = []
        
        # teacher EMA model
        with torch.no_grad():
            for aug_type in self.cfg.tta.aug_types.split(','):
                # Forward pass with ema model
                pc_aug, mat_aug = augment_cloud(batch_data['point_clouds'], type=aug_type)
                end_points_ema = self.model_ema({'point_clouds': pc_aug})
                grasp_preds_raw_ema, objectness_masks, grasp_angles, grasp_tolerances = pred_decode_raw_graspnet(end_points_ema)
                
                # get grasp preds
                if aug_type == 'none':
                    grasp_preds = [x[objectness_masks[b]] for b, x in enumerate(grasp_preds_raw_ema)]
                    features_ema.append(end_points_ema['seed_features'].clone()) # [B, 256, num_seed]
                    

                gg_array_filt = []
                for b in range(batch_size):

                    gg_array_ema = grasp_preds_raw_ema[b][objectness_masks[b]]
                    scene_cloud = batch_data['point_clouds_raw'][b]
                    scene_cloud = transform_point_cloud(scene_cloud, mat_aug)

                    if self.cfg.tta.geval_net.grasp_q_thresh > 0:
                        
                        # prepare object and gripper clouds
                        obj_clouds, gripper_clouds, gg_array_filt = crop_inner_cloud(
                            gg_array_ema.clone(), scene_cloud, self.cfg.tta.geval_net.min_points,
                        )
                        if len(obj_clouds) == 0: # no valid grasp predictions
                            print("No valid grasp predictions (crop_inner_cloud returned empty)")
                            return grasp_preds, None
                        
                        # run geval net in batch-wise manner
                        grasp_q = []
                        uncertainty = []
                        n_items = len(obj_clouds)
                        geval_bs = self.cfg.tta.geval_net.batch_size
                        for batch_idx in range(math.ceil(n_items / geval_bs)):
                            start_idx = batch_idx * geval_bs
                            end_idx = min(start_idx + geval_bs, n_items)
                            obj_clouds_batch = obj_clouds[start_idx:end_idx]
                            gripper_clouds_batch = gripper_clouds[start_idx:end_idx]

                            # use mc_dropout if enabled                            
                            if self.cfg.tta.geval_net.uncertainty_thresh > 0.0:
                                grasp_q_batch, std_batch = self.geval_net.forward_mc_dropout(obj_clouds_batch, gripper_clouds_batch, N=10)
                                uncertainty.extend(std_batch)
                            else:
                                grasp_q_batch = self.geval_net(obj_clouds_batch, gripper_clouds_batch)
                            grasp_q.extend(grasp_q_batch)
                            
                        grasp_q = torch.cat(grasp_q, dim=0)

                        if grasp_q.numel() == 0 or grasp_q.dim() == 0:
                            print("No valid grasp predictions (after grasp_q check)")
                            return grasp_preds, None
                        
                        if self.cfg.tta.geval_net.uncertainty_thresh > 0.0:
                            uncertainty = torch.cat(uncertainty, dim=0)
                            good_indices = torch.where(
                                (uncertainty < self.cfg.tta.geval_net.uncertainty_thresh) & \
                                 (grasp_q >= self.cfg.tta.geval_net.grasp_q_thresh))[0]
                        else:
                            good_indices = torch.where(grasp_q >= self.cfg.tta.geval_net.grasp_q_thresh)[0]
                        gg_array_filt = gg_array_filt[good_indices]
                        # # filter by grasp score threshold. then select top max_grasp_num grasps
                        # top_indices = torch.argsort(grasp_q, descending=True)[:self.cfg.tta.geval_net.max_grasp_num]
                        # grasp_q = grasp_q[top_indices]
                        # good_indices = top_indices[grasp_q >= self.cfg.tta.geval_net.grasp_q_thresh]
                        # gg_array_filt = gg_array_filt[good_indices]
                        
                    else:
                        gg_array_filt = gg_array_ema
                    if gg_array_filt.shape[0] == 0:
                        print("No valid grasp predictions (after grasp_q check2)")
                        return grasp_preds, None
                    grasp_preds_ensembles[b].append(gg_array_filt.clone())

                    #visualize gg_array_filt
                    # import open3d as o3d
                    # _gg = GraspGroup()
                    # _gg.grasp_group_array = gg_array_filt.cpu().numpy()
                    # cloud_o3d = o3d.geometry.PointCloud()
                    # cloud_o3d.points = o3d.utility.Vector3dVector(scene_cloud.cpu().numpy())
                    # o3d.visualization.draw_geometries([cloud_o3d] + _gg.to_open3d_geometry_list())

                    # After filtering, we need to handle EMA and analytical samples differently
                    idx_in_ema = torch.tensor(get_index_A_to_B(gg_array_filt.clone(), gg_array_ema.clone()), device=gg_array_ema.device)

                    # For original EMA predictions
                    original_indices = torch.where(objectness_masks[b])[0][idx_in_ema]
                    # For arrays that require exact indices from the original data
                    merged_grasp_preds_ema[b].append(grasp_preds_raw_ema[b][original_indices])
                    merged_grasp_top_view_inds_ema[b].append(end_points_ema['grasp_top_view_inds'][b][original_indices])
                    merged_grasp_angles_ema[b].append(grasp_angles[b][original_indices])
                    merged_grasp_tolerances_ema[b].append(grasp_tolerances[b][original_indices])
                    # Update merged_mat_aug for all filtered grasps
                    merged_mat_aug[b].extend([mat_aug] * gg_array_filt.shape[0])

            # merge all the grasp predictions from different augmentations
            for b in range(len(merged_grasp_preds_ema)):
                merged_grasp_preds_ema[b] = torch.cat(merged_grasp_preds_ema[b], dim=0)
                merged_grasp_top_view_inds_ema[b] = torch.cat(merged_grasp_top_view_inds_ema[b], dim=0)
                merged_grasp_angles_ema[b] = torch.cat(merged_grasp_angles_ema[b], dim=0)
                merged_grasp_tolerances_ema[b] = torch.cat(merged_grasp_tolerances_ema[b], dim=0)
                             
            batch_data['batch_grasp_preds_ema'] = merged_grasp_preds_ema # list of (Np', 17)
            batch_data['batch_grasp_top_view_inds_ema'] = merged_grasp_top_view_inds_ema # list of (Np',)
            batch_data['batch_grasp_angles_ema'] = merged_grasp_angles_ema # list of (Np', 1)
            batch_data['batch_grasp_tolerances_ema'] = merged_grasp_tolerances_ema # list of (Np', 1)
            batch_data['mat_aug'] = merged_mat_aug

        # merge all the grasp predictions from different ensembles
        grasp_preds = [np.zeros((0, 17)) for _ in range(batch_size)]
        for b in range(len(grasp_preds_ensembles)):
            for i in range(len(grasp_preds_ensembles[b])):
                gg = GraspGroup(grasp_preds_ensembles[b][i].cpu().numpy())
                H = np.eye(4)
                H[:3, :3] = get_aug_matrix(self.cfg.tta.aug_types.split(',')[i])
                gg = gg.transform(H)
                grasp_preds[b] = np.concatenate([grasp_preds[b], gg.grasp_group_array], axis=0)

        # student model
        end_points = self.model(batch_data)
        loss, end_points = compute_tta_loss(end_points)
             
            
        loss.backward()
        self.optimizer.step()

        # Update teacher model with EMA
        self.model_ema = ema_update_model(self.model_ema, self.model, self.cfg.tta.ema_ratio, self.device)
        # stochastic restore
        if self.cfg.tta.rst_ratio > 0:
            self.stochastic_restore(self.model, self.model_states)
        return grasp_preds, end_points
    

                        
                                               
class TTA_Grasp_EconomicGrasp(TTA_Grasp_Base):
    
    def __init__(self, cfg, model):
        super().__init__(cfg, model)



    def configure_model(self):
        
        super().configure_model()

        from economic_grasp.models.economicgrasp import load_economicgrasp, pred_decode_raw, pred_decode
        from economic_grasp.models.loss_economicgrasp import get_tta_loss 
        self.load_economicgrasp = load_economicgrasp
        self.pred_decode = pred_decode
        self.pred_decode_raw = pred_decode_raw
        self.compute_tta_loss = get_tta_loss

        self.model_ema = self.load_economicgrasp(deepcopy(self.cfg), self.device)
        self.model_ema.cfg.tta.method = "none" # to avoid processing labels for teacher EMA model
        self.model_ema.view.is_training = False
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)

        for param in self.model_ema.parameters():
            param.detach_()
        self.model_ema.to(self.device)
        self.model_ema.eval()


    def forward_and_adapt(self, batch_data):
        
        self.optimizer.zero_grad()
        batch_size = batch_data['point_clouds'].shape[0]
        merged_grasp_preds_ema = [[] for _ in range(batch_size)]
        merged_grasp_top_view_inds_ema = [[] for _ in range(batch_size)]
        merged_grasp_angles_ema = [[] for _ in range(batch_size)]
        merged_mat_aug = [[] for _ in range(batch_size)]
        grasp_preds_ensembles = [[] for _ in range(batch_size)]
        merged_grasp_q = [[] for _ in range(batch_size)]
        merged_view_scores_ema = [[] for _ in range(batch_size)]
        merged_graspness_scores_ema = [[] for _ in range(batch_size)]
        merged_objectness_scores_ema = [[] for _ in range(batch_size)]
        
        # teacher EMA model
        with torch.no_grad():
            for aug_type in self.cfg.tta.aug_types.split(','):
                # Forward pass with ema model
                pc_aug, mat_aug = augment_cloud(batch_data['point_clouds'], type=aug_type)
                
                end_points_ema = self.model_ema({
                    'point_clouds': pc_aug, 
                    'coordinates_for_voxel': pc_aug / self.cfg.model.voxel_size}
                    )
                # [1, 1024, 17], [1, 1024]
                grasp_preds_raw_ema, grasp_angles, view_scores, graspness_score, objectness_score = self.pred_decode_raw(end_points_ema)
                
                # get grasp preds
                if aug_type == 'none':
                    # grasp_preds = [x[objectness_masks[b]] for b, x in enumerate(grasp_preds_raw_ema)]
                    grasp_preds = grasp_preds_raw_ema
                    # features_ema.append(end_points_ema['seed_features'].clone()) # [B, 256, num_seed]
                    

                gg_array_filt = []
                for b in range(batch_size):

                    gg_array_ema = grasp_preds_raw_ema[b]
                    scene_cloud = batch_data['point_clouds_raw'][b]
                    scene_cloud = transform_point_cloud(scene_cloud, mat_aug)

                    if self.cfg.tta.geval_net.grasp_q_thresh > 0:
                        
                        # prepare object and gripper clouds
                        obj_clouds, gripper_clouds, gg_array_filt = crop_inner_cloud(
                            gg_array_ema.clone(), scene_cloud, self.cfg.tta.geval_net.min_points,
                        )
                        if len(obj_clouds) == 0: # no valid grasp predictions
                            print("No valid grasp predictions (crop_inner_cloud returned empty)")
                            return grasp_preds, None
                        
                        # run geval net in batch-wise manner
                        grasp_q = []
                        uncertainty = []
                        n_items = len(obj_clouds)
                        geval_bs = self.cfg.tta.geval_net.batch_size
                        for batch_idx in range(math.ceil(n_items / geval_bs)):
                            start_idx = batch_idx * geval_bs
                            end_idx = min(start_idx + geval_bs, n_items)
                            obj_clouds_batch = obj_clouds[start_idx:end_idx]
                            gripper_clouds_batch = gripper_clouds[start_idx:end_idx]

                            # use mc_dropout if enabled                            
                            if self.cfg.tta.geval_net.uncertainty_thresh > 0.0:
                                grasp_q_batch, std_batch = self.geval_net.forward_mc_dropout(
                                    obj_clouds_batch, gripper_clouds_batch, 
                                    N=self.cfg.tta.geval_net.uncertainty_n)
                                uncertainty.extend(std_batch)
                            else:
                                grasp_q_batch = self.geval_net(obj_clouds_batch, gripper_clouds_batch)
                            grasp_q.extend(grasp_q_batch)
                        grasp_q = torch.cat(grasp_q, dim=0)

                        if grasp_q.numel() == 0 or grasp_q.dim() == 0:
                            print("No valid grasp predictions (after grasp_q check)")
                            return grasp_preds, None
                        
                        if self.cfg.tta.geval_net.uncertainty_thresh > 0.0:
                            uncertainty = torch.cat(uncertainty, dim=0)
                            good_indices = torch.where(
                                (uncertainty < self.cfg.tta.geval_net.uncertainty_thresh) & \
                                 (grasp_q >= self.cfg.tta.geval_net.grasp_q_thresh))[0]
                        else:
                            good_indices = torch.where(grasp_q >= self.cfg.tta.geval_net.grasp_q_thresh)[0]
                        gg_array_filt = gg_array_filt[good_indices]
                        
                        grasp_q = grasp_q[good_indices]
                        merged_grasp_q[b].append(grasp_q.clone())
                        print(f"Grasp Q thresholding: {len(gg_array_filt)} valid grasps after filtering")
                        
                    else:
                        gg_array_filt = gg_array_ema
                        
                    if gg_array_filt.shape[0] == 0:
                        print("No valid grasp predictions (after grasp_q check2)")
                        return grasp_preds, None
                    grasp_preds_ensembles[b].append(gg_array_filt.clone())


                    # print(f'gg_array_filt shape: {gg_array_filt.shape}, gg_array_ema shape: {gg_array_ema.shape}')
                    # #visualize gg_array_filt
                    # import open3d as o3d
                    # _gg = GraspGroup()
                    # _gg.grasp_group_array = gg_array_filt.cpu().numpy()
                    # cloud_o3d = o3d.geometry.PointCloud()
                    # cloud_o3d.points = o3d.utility.Vector3dVector(scene_cloud.cpu().numpy())
                    # o3d.visualization.draw_geometries([cloud_o3d] + _gg.to_open3d_geometry_list())

                    # After filtering, we need to handle EMA and analytical samples differently
                    idx_in_ema = torch.tensor(get_index_A_to_B(gg_array_filt.clone(), gg_array_ema.clone()), device=gg_array_ema.device)
                    # For original EMA predictions
                    # For arrays that require exact indices from the original data
                    merged_grasp_preds_ema[b].append(grasp_preds_raw_ema[b][idx_in_ema])
                    merged_grasp_top_view_inds_ema[b].append(end_points_ema['grasp_top_view_inds'][b][idx_in_ema])
                    merged_grasp_angles_ema[b].append(grasp_angles[b][idx_in_ema])
                    merged_view_scores_ema[b].append(view_scores[b][idx_in_ema])
                    # valid_mask = torch.zeros_like(merged_grasp_angle_inds_ema[b])
                    # valid_mask[idx_in_ema] = 1
                    # Update merged_mat_aug for all filtered grasps
                    merged_mat_aug[b].extend([mat_aug] * gg_array_filt.shape[0])
                    merged_graspness_scores_ema[b].append(graspness_score)
                    merged_objectness_scores_ema[b].append(objectness_score)
                    

            # merge all the grasp predictions from different augmentations
            for b in range(len(merged_grasp_preds_ema)):
                merged_grasp_preds_ema[b] = torch.cat(merged_grasp_preds_ema[b], dim=0)
                merged_grasp_top_view_inds_ema[b] = torch.cat(merged_grasp_top_view_inds_ema[b], dim=0)
                merged_grasp_angles_ema[b] = torch.cat(merged_grasp_angles_ema[b], dim=0)
                merged_view_scores_ema[b] = torch.cat(merged_view_scores_ema[b], dim=0)
                # average graspness scores
                merged_graspness_scores_ema[b] = torch.stack(merged_graspness_scores_ema[b], dim=0).mean(dim=0)
                merged_objectness_scores_ema[b] = torch.stack(merged_objectness_scores_ema[b], dim=0).mean(dim=0)
            merged_graspness_scores_ema = torch.cat(merged_graspness_scores_ema, dim=0)
            merged_objectness_scores_ema = torch.cat(merged_objectness_scores_ema, dim=0)
                             
            batch_data['batch_grasp_preds_ema'] = merged_grasp_preds_ema # list of (Np', 17)
            batch_data['batch_grasp_top_view_inds_ema'] = merged_grasp_top_view_inds_ema # list of (Np',)
            batch_data['batch_grasp_angles_ema'] = merged_grasp_angles_ema # list of (Np', 1)
            batch_data['mat_aug'] = merged_mat_aug
            batch_data['batch_view_scores_ema'] = merged_view_scores_ema # list of (Np', 1)
            batch_data['graspness_label'] = merged_graspness_scores_ema # list of (Np', 1)
            batch_data['objectness_label'] = torch.argmax(merged_objectness_scores_ema, dim=1) # list of (Np', 2)
            
        
            
        # merge all the grasp predictions from different ensembles
        grasp_preds = [np.zeros((0, 17)) for _ in range(batch_size)]
        for b in range(len(grasp_preds_ensembles)):
            for i in range(len(grasp_preds_ensembles[b])):
                gg = GraspGroup(grasp_preds_ensembles[b][i].cpu().numpy())
                H = np.eye(4)
                H[:3, :3] = get_aug_matrix(self.cfg.tta.aug_types.split(',')[i])
                gg = gg.transform(H)
                grasp_preds[b] = np.concatenate([grasp_preds[b], gg.grasp_group_array], axis=0)
        
        num_grasps = len(grasp_preds[0])
            
        if num_grasps > self.cfg.tta.min_grasps:
            # student model
            end_points = self.model(batch_data)
            end_points['loss_type'] = self.cfg.tta.loss_type
            loss, end_points = self.compute_tta_loss(end_points)
                
            loss.backward()
            # clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Update teacher model with EMA
            self.model_ema = ema_update_model(self.model_ema, self.model, self.cfg.tta.ema_ratio, self.device)
            # stochastic restore
            if self.cfg.tta.rst_ratio > 0:
                self.stochastic_restore(self.model, self.model_states)
        else:
            end_points = {}
            
        # end_points = self.model_ema(batch_data)
        # grasp_preds, _, _, _, _ = self.pred_decode_raw(end_points)
        # self.model.eval()
        # self.model.view.is_training = False
        # end_points = self.model(batch_data)
        # grasp_preds = self.pred_decode(end_points)
        
        # self.model.train()
        # self.model.view.is_training = True

        return grasp_preds, end_points
    
    def stochastic_restore(self, model, model_states):
        for nm, m in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.cfg.tta.rst_ratio).float().cuda()
                    with torch.no_grad():
                        p.data = model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)
                        
                        