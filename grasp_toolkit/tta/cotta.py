import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from graspnetAPI import GraspGroup

from .base import TTA_Base
from copy import deepcopy

# from graspnet_baseline.models.graspnet import pred_decode, pred_decode_raw
# from graspnet_baseline.models.loss import compute_tta_loss

from grasp_toolkit.utils import (
    ema_update_model, 
    get_aug_matrix,
    augment_cloud, 
    load_optimizer
)


class CoTTA(TTA_Base):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

    def configure_model(self):
        # initialize ema model
        self.model_ema = deepcopy(self.model)
        self.device = next(self.model.parameters()).device
        self.model_ema.to(self.device)
        self.model_ema.eval()
        self.model_ema.requires_grad_(False)
        self.model_ema.cfg.tta.method = "none" # disable TTA for ema model
        self.model_ema.grasp_generator.cfg.tta.method = "none"

        self.model_states = [deepcopy(model.state_dict()) for model in self.model.modules()]
        self.model.eval()
        # disable grad to enable only what we need
        self.model.requires_grad_(False)
        for m in self.model.modules():
            m.requires_grad_(True) # enable grad for all modules
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        self.optimizer = load_optimizer(self.cfg.tta.optimizer, self.model, self.cfg.tta.lr)


    @torch.enable_grad()
    def forward_and_adapt(self, batch_data):

        self.optimizer.zero_grad()
        batch_size = batch_data['point_clouds'].shape[0]
        merged_grasp_preds_ema = [[] for _ in range(batch_size)]
        merged_grasp_top_view_inds_ema = [[] for _ in range(batch_size)]
        merged_grasp_angles_ema = [[] for _ in range(batch_size)]
        merged_grasp_tolerances_ema = [[] for _ in range(batch_size)]
        merged_mat_aug = [[] for _ in range(batch_size)]
        grasp_preds_ensembles = [[] for _ in range(batch_size)]

        # teacher EMA model
        with torch.no_grad():
            for aug_type in self.cfg.tta.aug_types.split(','):
                # Forward pass with ema model
                pc_aug, mat_aug = augment_cloud(batch_data['point_clouds'], type=aug_type)
                end_points_ema = self.model_ema({'point_clouds': pc_aug})
                grasp_preds_raw_ema, objectness_masks, grasp_angles, grasp_tolerances = pred_decode_raw(end_points_ema)
                if aug_type == 'none':
                    grasp_preds = pred_decode(end_points_ema)
                # iterate over batches
                for b in range(batch_size):
                    merged_grasp_preds_ema[b].append(grasp_preds_raw_ema[b][objectness_masks[b]])
                    merged_grasp_top_view_inds_ema[b].append(end_points_ema['grasp_top_view_inds'][b][objectness_masks[b]])
                    merged_grasp_angles_ema[b].append(grasp_angles[b][objectness_masks[b]])
                    merged_grasp_tolerances_ema[b].append(grasp_tolerances[b][objectness_masks[b]])
                    merged_mat_aug[b].extend([mat_aug] * grasp_preds_raw_ema[b][objectness_masks[b]].shape[0])
                    grasp_preds_ensembles[b] = merged_grasp_preds_ema[b]

            # merge all the grasp predictions from different augmentations
            for b in range(len(merged_grasp_preds_ema)):
                merged_grasp_preds_ema[b] = torch.cat(merged_grasp_preds_ema[b], dim=0)
                merged_grasp_top_view_inds_ema[b] = torch.cat(merged_grasp_top_view_inds_ema[b], dim=0)
                merged_grasp_angles_ema[b] = torch.cat(merged_grasp_angles_ema[b], dim=0)
                merged_grasp_tolerances_ema[b] = torch.cat(merged_grasp_tolerances_ema[b], dim=0)
                # filter by grasp score threshold  
                if self.cfg.tta.grasp_score_thresh > 0:
                    scores_mask = merged_grasp_preds_ema[b][:, 0] >= self.cfg.tta.grasp_score_thresh
                    if scores_mask.sum() == 0:
                        return grasp_preds, None
                    merged_grasp_preds_ema[b] = merged_grasp_preds_ema[b][scores_mask]
                    merged_grasp_top_view_inds_ema[b] = merged_grasp_top_view_inds_ema[b][scores_mask]
                    merged_grasp_angles_ema[b] = merged_grasp_angles_ema[b][scores_mask]
                    merged_grasp_tolerances_ema[b] = merged_grasp_tolerances_ema[b][scores_mask]
                    merged_mat_aug[b] = [merged_mat_aug[b][i] for i in range(len(merged_mat_aug[b])) if scores_mask[i]]

            batch_data['batch_grasp_preds_ema'] = merged_grasp_preds_ema # list of (Np', 17)
            batch_data['batch_grasp_top_view_inds_ema'] = merged_grasp_top_view_inds_ema # list of (Np',)
            batch_data['batch_grasp_angles_ema'] = merged_grasp_angles_ema # list of (Np', 1)
            batch_data['batch_grasp_tolerances_ema'] = merged_grasp_tolerances_ema # list of (Np', 1)
            batch_data['mat_aug'] = merged_mat_aug
        
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

        # torch.cuda.empty_cache()
        return grasp_preds, end_points  
    
    def stochastic_restore(self, model, model_states):
        for nm, m in model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape) < self.cfg.tta.rst_ratio).float().cuda()
                    with torch.no_grad():
                        p.data = model_states[0][f"{nm}.{npp}"] * mask + p * (1.-mask)