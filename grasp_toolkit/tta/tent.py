import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import TTA_Base

from grasp_toolkit.utils.tta_utils import (
    load_optimizer
)

def entropy(predictions: torch.Tensor, class_dim: int) -> torch.Tensor:
    """
    Calculates the mean entropy of predictions.
    Entropy is calculated for each element over the class_dim,
    then averaged over all other dimensions (batch, spatial, etc.).
    """
    probs = F.softmax(predictions, dim=class_dim)
    log_probs = torch.log(probs + 1e-9)  # Add epsilon for numerical stability
    # Calculate entropy for each prediction distribution
    # Summing over the class dimension: H(p) = - sum(p_i * log(p_i))
    element_entropy = -torch.sum(probs * log_probs, dim=class_dim)
    # Return the mean entropy over all other dimensions
    return element_entropy.mean()

# class Entropy(nn.Module):
#     def __init__(self):
#         super(Entropy, self).__init__()

#     def __call__(self, logits):
#         return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)

class TENT(TTA_Base):
    def __init__(self, cfg, model):
        super().__init__(cfg, model)

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False) # disable grad, to (re-)enable only what tent updates
        for m in self.model.modules():
            # configure norm for tent updates: enable grad + force batch statisics
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
                m.requires_grad_(True)
        self.optimizer = load_optimizer(self.cfg.tta.optimizer, self.model, self.cfg.tta.lr, self.cfg.tta.backbone_lr_ratio)


    @torch.enable_grad()
    def forward_and_adapt(self, batch_data):
        
        end_points = self.model(batch_data)
        # grasp_preds = self.pred_decode(end_points)
        total_loss = 0
        if self.cfg.model.name == 'economic_grasp':
            total_loss += entropy(end_points['grasp_score_pred'], class_dim=1)  # Assuming grasp_score_pred is the logits for depth bins
            total_loss += entropy(end_points['grasp_angle_pred'], class_dim=1)  # Assuming grasp_angle_cls_pred is the logits for angle classes
            total_loss += entropy(end_points['grasp_depth_pred'], class_dim=1)  # Assuming objectness_score is the logits for object/non-object classes
            total_loss += entropy(end_points['grasp_width_pred'], class_dim=1)  # Assuming objectness_score is the logits for object/non-object classes
        
        # if 'objectness' in self.cfg.tta.target:
        #     # end_points['objectness_score'] shape: (B, 2, Ns)
        #     # B: batch_size, 2: object/non-object classes, Ns: num_seed_points
        #     objectness_logits = end_points['objectness_score']
        #     loss_obj = entropy(objectness_logits, class_dim=1) # Softmax over class dim 1
        #     total_loss += loss_obj
        #     # print(f"Objectness loss: {loss_obj.item()}") # For debugging

        # if 'grasp_angle' in self.cfg.tta.target:
        #     # end_points['grasp_angle_cls_pred'] shape: (B, A, Ns, D)
        #     # A: num_angle_classes, D: num_depth_bins
        #     angle_logits = end_points['grasp_angle_cls_pred']
        #     loss_angle = entropy(angle_logits, class_dim=1) # Softmax over angle class dim 1 (A)
        #     total_loss += loss_angle
        #     # print(f"Grasp angle loss: {loss_angle.item()}") # For debugging

        # if 'grasp_depth' in self.cfg.tta.target:
        #     # end_points['grasp_score_pred'] shape: (B, A, Ns, D)
        #     # These are often regression targets or scores not directly for depth classification.
        #     # However, if interpreted as logits for depth bins *for each angle and point*:
        #     # We want to make depth prediction confident over the D dimension.
        #     depth_logits = end_points['grasp_score_pred'] 
        #     loss_depth = entropy(depth_logits, class_dim=3) # Softmax over depth dim 3 (D)
        #     total_loss += loss_depth
        #     # print(f"Grasp depth loss: {loss_depth.item()}") # For debugging
            
        # loss = entropy(torch.cat(grasp_score_list, 0).reshape(-1))
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        grasp_preds = self.pred_decode(end_points)

        grasp_preds = [grasp_preds[i].detach() for i in range(len(grasp_preds))]
        return grasp_preds, end_points